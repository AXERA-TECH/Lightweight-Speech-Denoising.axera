import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import copy


def get_opset_version(model):
    for opset in model.opset_import:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            return opset.version
    return 11  # 默认返回11


def get_initializer_value(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def get_shape(model, tensor_name):
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            shape = []
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            return shape
    
    for inp in model.graph.input:
        if inp.name == tensor_name:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            return shape
    
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)
    
    return None


def is_consecutive_indices(indices):
    if len(indices) < 1:
        return False, None, None
    
    indices = np.array(indices).flatten()
    start = int(indices[0])
    expected = np.arange(start, start + len(indices))
    
    if np.array_equal(indices, expected):
        return True, start, start + len(indices)
    return False, None, None


def analyze_multidim_indices(indices):
    if indices.ndim != 2:
        return False, None
    
    num_rows = indices.shape[0]
    row_ranges = []
    
    for i in range(num_rows):
        row = indices[i]
        is_consec, start, end = is_consecutive_indices(row)
        if not is_consec:
            return False, None
        row_ranges.append((start, end))
    
    return True, row_ranges


def create_unique_name(base_name, existing_names):
    if base_name not in existing_names:
        existing_names.add(base_name)
        return base_name
    
    counter = 0
    while f"{base_name}_{counter}" in existing_names:
        counter += 1
    
    new_name = f"{base_name}_{counter}"
    existing_names.add(new_name)
    return new_name


def analyze_gather_nodes(model):
    gather_info = []
    
    for node in model.graph.node:
        if node.op_type == 'Gather':
            info = {
                'node': node,
                'name': node.name,
                'input_data': node.input[0],
                'input_indices': node.input[1],
                'output': node.output[0],
                'axis': 0
            }
            
            for attr in node.attribute:
                if attr.name == 'axis':
                    info['axis'] = attr.i
            
            indices_value = get_initializer_value(model, node.input[1])
            if indices_value is not None:
                info['indices_value'] = indices_value
                info['indices_shape'] = indices_value.shape
                
                if indices_value.ndim == 0 or (indices_value.ndim == 1 and indices_value.size == 1):
                    idx = int(indices_value.flatten()[0])
                    info['type'] = 'scalar'
                    info['scalar_index'] = idx
                    
                elif indices_value.ndim == 1:
                    # 一维数组
                    is_consec, start, end = is_consecutive_indices(indices_value)
                    if is_consec:
                        info['type'] = 'consecutive_1d'
                        info['start'] = start
                        info['end'] = end
                    else:
                        info['type'] = 'general_1d'
                        
                elif indices_value.ndim == 2:
                    is_all_consec, row_ranges = analyze_multidim_indices(indices_value)
                    if is_all_consec:
                        info['type'] = 'consecutive_2d'
                        info['row_ranges'] = row_ranges
                        info['num_rows'] = indices_value.shape[0]
                        info['row_length'] = indices_value.shape[1]
                    else:
                        info['type'] = 'general_2d'
                else:
                    info['type'] = 'general_nd'
            else:
                info['type'] = 'dynamic'

            input_shape = get_shape(model, info['input_data'])
            info['input_shape'] = input_shape
            
            gather_info.append(info)
    
    return gather_info


def optimize_gather_nodes(model):
    model = copy.deepcopy(model)
    
    opset_version = get_opset_version(model)
    print(f"  检测到opset版本: {opset_version}")
    
    use_axes_as_input = opset_version >= 13
    
    existing_names = set()
    for node in model.graph.node:
        existing_names.add(node.name)
        for out in node.output:
            existing_names.add(out)
    for init in model.graph.initializer:
        existing_names.add(init.name)
    
    gather_info = analyze_gather_nodes(model)
    
    replacements = {}
    initializers_to_add = []
    initializers_to_remove = []
    
    optimization_stats = {
        'scalar': 0,
        'consecutive_1d': 0,
        'consecutive_2d': 0,
        'skipped': 0
    }
    
    node_to_idx = {id(node): idx for idx, node in enumerate(model.graph.node)}
    
    for info in gather_info:
        node = info['node']
        axis = info['axis']
        node_idx = node_to_idx[id(node)]
        new_nodes = []
        
        if info['type'] == 'scalar':
            idx = info['scalar_index']
            optimization_stats['scalar'] += 1
            
            slice_starts_name = create_unique_name(f"{node.name}_starts", existing_names)
            slice_ends_name = create_unique_name(f"{node.name}_ends", existing_names)
            slice_axes_name = create_unique_name(f"{node.name}_axes", existing_names)
            slice_output_name = create_unique_name(f"{node.name}_slice_out", existing_names)
            
            starts_tensor = numpy_helper.from_array(np.array([idx], dtype=np.int64), slice_starts_name)
            ends_tensor = numpy_helper.from_array(np.array([idx + 1], dtype=np.int64), slice_ends_name)
            axes_tensor = numpy_helper.from_array(np.array([axis], dtype=np.int64), slice_axes_name)
            
            initializers_to_add.extend([starts_tensor, ends_tensor, axes_tensor])
            
            slice_node = helper.make_node(
                'Slice',
                inputs=[info['input_data'], slice_starts_name, slice_ends_name, slice_axes_name],
                outputs=[slice_output_name],
                name=create_unique_name(f"{node.name}_slice", existing_names)
            )
            
            if use_axes_as_input:
                squeeze_axes_name = create_unique_name(f"{node.name}_squeeze_axes", existing_names)
                squeeze_axes_tensor = numpy_helper.from_array(np.array([axis], dtype=np.int64), squeeze_axes_name)
                initializers_to_add.append(squeeze_axes_tensor)
                squeeze_node = helper.make_node(
                    'Squeeze',
                    inputs=[slice_output_name, squeeze_axes_name],
                    outputs=[info['output']],
                    name=create_unique_name(f"{node.name}_squeeze", existing_names)
                )
            else:
                squeeze_node = helper.make_node(
                    'Squeeze',
                    inputs=[slice_output_name],
                    outputs=[info['output']],
                    name=create_unique_name(f"{node.name}_squeeze", existing_names),
                    axes=[axis]
                )
            
            new_nodes = [slice_node, squeeze_node]
            initializers_to_remove.append(info['input_indices'])
            
        elif info['type'] == 'consecutive_1d':
            optimization_stats['consecutive_1d'] += 1
            
            start = info['start']
            end = info['end']
            
            slice_starts_name = create_unique_name(f"{node.name}_starts", existing_names)
            slice_ends_name = create_unique_name(f"{node.name}_ends", existing_names)
            slice_axes_name = create_unique_name(f"{node.name}_axes", existing_names)
            
            starts_tensor = numpy_helper.from_array(np.array([start], dtype=np.int64), slice_starts_name)
            ends_tensor = numpy_helper.from_array(np.array([end], dtype=np.int64), slice_ends_name)
            axes_tensor = numpy_helper.from_array(np.array([axis], dtype=np.int64), slice_axes_name)
            
            initializers_to_add.extend([starts_tensor, ends_tensor, axes_tensor])
            
            slice_node = helper.make_node(
                'Slice',
                inputs=[info['input_data'], slice_starts_name, slice_ends_name, slice_axes_name],
                outputs=[info['output']],
                name=create_unique_name(f"{node.name}_slice", existing_names)
            )
            
            new_nodes = [slice_node]
            initializers_to_remove.append(info['input_indices'])
            
        elif info['type'] == 'consecutive_2d':
            optimization_stats['consecutive_2d'] += 1
            
            num_rows = info['num_rows']
            row_ranges = info['row_ranges']
            
            unsqueeze_output = create_unique_name(f"{node.name}_unsqueezed", existing_names)
            
            if use_axes_as_input:
                unsqueeze_axes_name = create_unique_name(f"{node.name}_unsqueeze_axes", existing_names)
                unsqueeze_axes_tensor = numpy_helper.from_array(np.array([axis], dtype=np.int64), unsqueeze_axes_name)
                initializers_to_add.append(unsqueeze_axes_tensor)
                unsqueeze_node = helper.make_node(
                    'Unsqueeze',
                    inputs=[info['input_data'], unsqueeze_axes_name],
                    outputs=[unsqueeze_output],
                    name=create_unique_name(f"{node.name}_unsqueeze", existing_names)
                )
            else:
                unsqueeze_node = helper.make_node(
                    'Unsqueeze',
                    inputs=[info['input_data']],
                    outputs=[unsqueeze_output],
                    name=create_unique_name(f"{node.name}_unsqueeze", existing_names),
                    axes=[axis]
                )
            new_nodes.append(unsqueeze_node)
            
            slice_outputs = []
            new_axis = axis + 1
            
            for i, (start, end) in enumerate(row_ranges):
                slice_starts_name = create_unique_name(f"{node.name}_row{i}_starts", existing_names)
                slice_ends_name = create_unique_name(f"{node.name}_row{i}_ends", existing_names)
                slice_axes_name = create_unique_name(f"{node.name}_row{i}_axes", existing_names)
                slice_output = create_unique_name(f"{node.name}_row{i}_slice", existing_names)
                
                starts_tensor = numpy_helper.from_array(np.array([start], dtype=np.int64), slice_starts_name)
                ends_tensor = numpy_helper.from_array(np.array([end], dtype=np.int64), slice_ends_name)
                axes_tensor = numpy_helper.from_array(np.array([new_axis], dtype=np.int64), slice_axes_name)
                
                initializers_to_add.extend([starts_tensor, ends_tensor, axes_tensor])
                
                slice_node = helper.make_node(
                    'Slice',
                    inputs=[unsqueeze_output, slice_starts_name, slice_ends_name, slice_axes_name],
                    outputs=[slice_output],
                    name=create_unique_name(f"{node.name}_row{i}_slice_op", existing_names)
                )
                new_nodes.append(slice_node)
                slice_outputs.append(slice_output)
            
            concat_node = helper.make_node(
                'Concat',
                inputs=slice_outputs,
                outputs=[info['output']],
                name=create_unique_name(f"{node.name}_concat", existing_names),
                axis=axis
            )
            new_nodes.append(concat_node)
            
            initializers_to_remove.append(info['input_indices'])
            
        else:
            optimization_stats['skipped'] += 1
            print(f"  跳过节点 {node.name}: 类型={info['type']}, 无法优化")
            continue
        
        if new_nodes:
            replacements[node_idx] = new_nodes
    
    new_node_list = []
    for idx, node in enumerate(model.graph.node):
        if idx in replacements:
            new_node_list.extend(replacements[idx])
        else:
            new_node_list.append(node)
    
    while len(model.graph.node) > 0:
        model.graph.node.pop()
    
    for node in new_node_list:
        model.graph.node.append(node)
    
    model.graph.initializer.extend(initializers_to_add)
    
    initializers_to_remove_set = set(initializers_to_remove)
    used_inputs = set()
    for node in model.graph.node:
        for inp in node.input:
            used_inputs.add(inp)
    
    for init in list(model.graph.initializer):
        if init.name in initializers_to_remove_set and init.name not in used_inputs:
            model.graph.initializer.remove(init)
    
    return model, optimization_stats


def print_gather_analysis(model):
    gather_info = analyze_gather_nodes(model)
    
    print(f"\n{'='*70}")
    print(f"找到 {len(gather_info)} 个Gather节点")
    print(f"{'='*70}\n")
    
    for i, info in enumerate(gather_info):
        print(f"[{i+1}] 节点名: {info['name']}")
        print(f"    输入数据: {info['input_data']}")
        if info.get('input_shape'):
            print(f"    输入Shape: {info['input_shape']}")
        print(f"    输入索引: {info['input_indices']}")
        print(f"    输出: {info['output']}")
        print(f"    Axis: {info['axis']}")
        
        if 'indices_value' in info:
            indices = info['indices_value']
            print(f"    Indices Shape: {indices.shape}")
            
            if indices.size <= 20:
                if indices.ndim <= 1:
                    print(f"    Indices值: {indices.flatten().tolist()}")
                else:
                    print(f"    Indices值:")
                    for row_idx in range(min(indices.shape[0], 5)):
                        row = indices[row_idx]
                        if len(row) <= 10:
                            print(f"      Row {row_idx}: {row.tolist()}")
                        else:
                            print(f"      Row {row_idx}: [{row[0]}, {row[1]}, ..., {row[-2]}, {row[-1]}]")
            else:
                if indices.ndim == 2:
                    print(f"    Indices值 (每行首尾):")
                    for row_idx in range(min(indices.shape[0], 5)):
                        row = indices[row_idx]
                        print(f"      Row {row_idx}: [{row[0]}, {row[1]}, ..., {row[-2]}, {row[-1]}]")
                    if indices.shape[0] > 5:
                        print(f"      ... 共 {indices.shape[0]} 行")
                else:
                    print(f"    Indices前几个值: {indices.flatten()[:10].tolist()}...")
            
            print(f"    分析类型: {info['type']}")
            
            if info['type'] == 'scalar':
                print(f"    ✓ 可优化为: Slice[{info['scalar_index']}:{info['scalar_index']+1}] + Squeeze")
            elif info['type'] == 'consecutive_1d':
                print(f"    ✓ 可优化为: Slice[{info['start']}:{info['end']}]")
            elif info['type'] == 'consecutive_2d':
                print(f"    ✓ 可优化为: Unsqueeze + {info['num_rows']}个Slice + Concat")
                print(f"      各行范围: {info['row_ranges'][:5]}{'...' if len(info['row_ranges']) > 5 else ''}")
            else:
                print(f"    ✗ 无法优化 (非连续索引或动态索引)")
        else:
            print(f"    类型: 动态索引 (无法优化)")
        
        print()
    
    return gather_info


def clear_gather(input_path, output_path=None):
    print(f"加载模型: {input_path}")
    model = onnx.load(input_path)
    
    print("\n" + "="*70)
    print("步骤1: 分析原始模型中的Gather节点")
    print("="*70)
    print_gather_analysis(model)
    
    print("\n" + "="*70)
    print("步骤2: 优化Gather节点")
    print("="*70)
    optimized_model, stats = optimize_gather_nodes(model)
    
    print(f"\n优化统计:")
    print(f"  - 标量索引 Gather (-> Slice + Squeeze): {stats['scalar']}")
    print(f"  - 1D连续索引 Gather (-> Slice): {stats['consecutive_1d']}")
    print(f"  - 2D连续索引 Gather (-> Unsqueeze + Slices + Concat): {stats['consecutive_2d']}")
    print(f"  - 跳过/无法优化: {stats['skipped']}")
    total_optimized = stats['scalar'] + stats['consecutive_1d'] + stats['consecutive_2d']
    print(f"  - 总共优化: {total_optimized}")
    
    print("\n" + "="*70)
    print("步骤3: 验证优化后的模型")
    print("="*70)
    
    try:
        onnx.checker.check_model(optimized_model)
        print("✓ 模型验证通过!")
    except Exception as e:
        print(f"✗ 模型验证失败: {e}")
        print("尝试跳过验证继续保存...")
    
    if output_path is None:
        output_path = input_path.replace('.onnx', '_optimized.onnx')
    
    print(f"\n保存优化后的模型到: {output_path}")
    onnx.save(optimized_model, output_path)
    
    remaining_gathers = [n for n in optimized_model.graph.node if n.op_type == 'Gather']
    if remaining_gathers:
        print(f"\n注意: 优化后仍有 {len(remaining_gathers)} 个Gather节点未能优化")
        for g in remaining_gathers:
            print(f"  - {g.name}")
    else:
        print("\n✓ 所有Gather节点已成功优化!")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python optimize_gather.py <input.onnx> [output.onnx]")
        print("\n请上传ONNX模型文件...")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        clear_gather(input_path, output_path)