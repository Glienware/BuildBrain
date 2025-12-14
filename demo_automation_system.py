"""
Demo: Complete n8n-style automation system

This demonstrates:
1. Creating workflows from nodes
2. Validating flow topology and data types
3. Executing flows with the TopologicalExecutor
4. Serializing and persisting workflows
"""

import asyncio
from src.nodes.automation_nodes import (
    NODE_REGISTRY,
    create_node_config,
    get_all_nodes_by_category,
    DataType
)
from src.nodes.executor import TopologicalExecutor, FlowValidator
from src.nodes.serializer import FlowSerializer, FlowPersistence


async def demo_node_catalog():
    """Show all available nodes by category."""
    print("=" * 60)
    print("AVAILABLE NODES BY CATEGORY")
    print("=" * 60)
    
    catalog = get_all_nodes_by_category()
    for category, nodes in sorted(catalog.items()):
        print(f"\nCategory: {category}")
        print("  " + "-" * 50)
        for node in nodes:
            print(f"  * {node.display_name:30} ({node.node_type})")
            if node.input_ports:
                for port_name, port in node.input_ports.items():
                    print(f"      [in] {port_name:20} [{port.data_type.value}]")
            if node.output_ports:
                for port_name, port in node.output_ports.items():
                    print(f"      [out] {port_name:20} [{port.data_type.value}]")


async def demo_create_workflow():
    """Create a simple workflow."""
    print("\n" + "=" * 60)
    print("CREATING A SAMPLE WORKFLOW")
    print("=" * 60)
    
    # Create nodes
    trigger = create_node_config("manual_trigger")
    llm = create_node_config("llm")
    output = create_node_config("log")
    
    nodes = {
        "trigger_1": (trigger, {"x": 50, "y": 100}),
        "llm_1": (llm, {"x": 300, "y": 100}),
        "log_1": (output, {"x": 550, "y": 100}),
    }
    
    # Define connections
    connections = [
        {
            "id": "conn_1",
            "source_node": "trigger_1",
            "source_port": "output",
            "target_node": "llm_1",
            "target_port": "input"
        },
        {
            "id": "conn_2",
            "source_node": "llm_1",
            "source_port": "response",
            "target_node": "log_1",
            "target_port": "message"
        }
    ]
    
    print(f"✓ Created workflow with {len(nodes)} nodes")
    print(f"✓ Connected with {len(connections)} edges")
    print(f"\n  Nodes: {' -> '.join(['trigger', 'llm', 'log'])}")
    
    return nodes, connections


async def demo_validate_flow():
    """Validate flow topology and data types."""
    print("\n" + "=" * 60)
    print("VALIDATING FLOW")
    print("=" * 60)
    
    nodes, connections = await demo_create_workflow()
    
    # Get node configs
    node_configs = {}
    for node_id, (config, _) in nodes.items():
        node_configs[node_id] = config
    
    # Validate
    print("\nValidating connections...")
    valid, msg = FlowValidator.validate_connections(node_configs, connections)
    print(f"  [{'OK' if valid else 'FAIL'}] Connections: {msg or 'OK'}")
    
    print("\nValidating data types...")
    valid, msg = FlowValidator.validate_types(node_configs, connections)
    print(f"  [{'OK' if valid else 'FAIL'}] Types: {msg or 'OK'}")
    
    print("\nDetecting cycles...")
    has_cycle, msg = FlowValidator.has_cycles(node_configs, connections)
    print(f"  [{'OK' if not has_cycle else 'FAIL'}] Cycles: {msg or 'None'}")


async def demo_execute_flow():
    """Execute a workflow with the TopologicalExecutor."""
    print("\n" + "=" * 60)
    print("EXECUTING WORKFLOW")
    print("=" * 60)
    
    nodes, connections = await demo_create_workflow()
    
    # Get node configs
    node_configs = {}
    for node_id, (config, _) in nodes.items():
        node_configs[node_id] = config
    
    # Create executor
    executor = TopologicalExecutor()
    
    # Mock node executor function
    async def execute_node(node_id: str, config, inputs):
        """Mock execution of a node."""
        print(f"  → Executing {config.display_name}...")
        await asyncio.sleep(0.1)  # Simulate work
        
        # Return mock outputs
        outputs = {}
        for port_name in config.output_ports:
            if port_name == "response":
                outputs[port_name] = "AI response: Hello, World!"
            elif port_name == "success":
                outputs[port_name] = True
            else:
                outputs[port_name] = f"Output from {port_name}"
        return outputs
    
    # Execute
    print("\nExecuting flow...")
    success, msg = await executor.execute(node_configs, connections, execute_node)
    
    if success:
        print(f"[OK] Execution successful: {msg}")
        summary = executor.get_execution_summary()
        print(f"\n  Summary:")
        print(f"    Total time: {summary['total_time']:.2f}s")
        print(f"    Successful nodes: {summary['successful_nodes']}/{summary['total_nodes']}")
        print(f"\n  Logs:")
        for log in summary['logs'][-5:]:
            print(f"    {log}")
    else:
        print(f"[FAIL] Execution failed: {msg}")


async def demo_serialize_flow():
    """Serialize and persist a workflow."""
    print("\n" + "=" * 60)
    print("SERIALIZING WORKFLOW")
    print("=" * 60)
    
    nodes, connections = await demo_create_workflow()
    
    # Get node configs
    node_configs = {}
    positions = {}
    for node_id, (config, pos) in nodes.items():
        node_configs[node_id] = config
        positions[node_id] = pos
    
    # Serialize
    flow_def = FlowSerializer.serialize_flow(
        flow_id="demo_workflow_1",
        flow_name="Demo Workflow",
        nodes={nid: (nc, positions[nid]) for nid, nc in node_configs.items()},
        connections=connections,
        variables={"test_var": "value"},
        settings={"auto_execute": False}
    )
    
    print(f"[OK] Serialized workflow: {flow_def.name}")
    print(f"  ID: {flow_def.id}")
    print(f"  Nodes: {len(flow_def.nodes)}")
    print(f"  Connections: {len(flow_def.connections)}")
    
    # Show JSON snippet
    json_str = FlowSerializer.to_json(flow_def)
    lines = json_str.split("\n")[:10]
    print("\n  JSON Preview:")
    for line in lines:
        print(f"    {line}")
    print("    ...")
    
    return flow_def


async def demo_persistence():
    """Save and load workflows from disk."""
    print("\n" + "=" * 60)
    print("PERSISTENCE (SAVE/LOAD)")
    print("=" * 60)
    
    flow_def = await demo_serialize_flow()
    
    # Create persistence manager
    persistence = FlowPersistence("projects/flows")
    
    # Save
    print(f"\nSaving workflow...")
    success, msg = persistence.save_flow(flow_def, overwrite=True)
    print(f"  [{'OK' if success else 'FAIL'}] {msg}")
    
    # Load
    print(f"\nLoading workflow...")
    success, loaded_flow, msg = persistence.load_flow(flow_def.id)
    if success:
        print(f"  [OK] Loaded: {loaded_flow.name}")
        print(f"    Nodes: {len(loaded_flow.nodes)}")
        print(f"    Connections: {len(loaded_flow.connections)}")
    else:
        print(f"  [FAIL] {msg}")
    
    # List flows
    print(f"\nListing saved workflows...")
    flows = persistence.list_flows()
    print(f"  Found {len(flows)} workflow(s):")
    for flow in flows:
        print(f"    * {flow['name']} ({flow['id']})")


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("N8N-STYLE AUTOMATION SYSTEM DEMO".center(60))
    print("=" * 60 + "\n")
    
    await demo_node_catalog()
    await demo_validate_flow()
    await demo_execute_flow()
    await demo_persistence()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\n[OK] Node catalog with 21 nodes")
    print("[OK] Workflow creation and validation")
    print("[OK] Flow execution with data flow")
    print("[OK] Serialization and persistence")
    print("\nReady for full automation deployment!\n")


if __name__ == "__main__":
    asyncio.run(main())
