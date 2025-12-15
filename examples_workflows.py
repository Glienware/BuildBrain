#!/usr/bin/env python3
"""
Ejemplo: Cómo cargar y ejecutar flujos .BUILDBM programáticamente
"""

import json
from pathlib import Path
from src.nodes.serializer import FlowPersistence
from src.nodes.executor import TopologicalExecutor, FlowValidator


def load_and_display_workflow(workflow_path: str):
    """Cargar un workflow y mostrar su estructura."""
    print(f"\n{'='*70}")
    print(f"Cargando: {workflow_path}")
    print(f"{'='*70}\n")
    
    persistence = FlowPersistence()
    success, agent_data, msg = persistence.load_agent(workflow_path)
    
    if not success:
        print(f"❌ Error: {msg}")
        return
    
    # Mostrar info general
    print(f"✓ Nombre: {agent_data.get('name')}")
    print(f"✓ Descripción: {agent_data.get('description')}")
    print(f"✓ Versión: {agent_data.get('version')}")
    
    # Mostrar nodos
    nodes = agent_data.get('nodes', {})
    print(f"\n📍 Nodos ({len(nodes)}):")
    for node_id, node_info in nodes.items():
        print(f"  • {node_id}: {node_info.get('display_name')} ({node_info.get('type')})")
    
    # Mostrar conexiones
    connections = agent_data.get('connections', [])
    print(f"\n🔗 Conexiones ({len(connections)}):")
    for conn in connections:
        src = conn.get('source_node')
        tgt = conn.get('target_node')
        src_port = conn.get('source_port')
        tgt_port = conn.get('target_port')
        print(f"  • {src}.{src_port} → {tgt}.{tgt_port}")
    
    # Mostrar variables
    variables = agent_data.get('variables', {})
    if variables:
        print(f"\n📦 Variables:")
        for var_name, var_value in variables.items():
            print(f"  • {var_name}: {var_value}")
    
    print()


def validate_workflow(workflow_path: str):
    """Validar estructura de un workflow."""
    print(f"\n{'='*70}")
    print(f"Validando: {workflow_path}")
    print(f"{'='*70}\n")
    
    persistence = FlowPersistence()
    success, agent_data, msg = persistence.load_agent(workflow_path)
    
    if not success:
        print(f"❌ Error de carga: {msg}")
        return False
    
    try:
        # Validar estructura mínima
        if 'nodes' not in agent_data:
            print("❌ Falta: 'nodes'")
            return False
        
        if 'connections' not in agent_data:
            print("❌ Falta: 'connections'")
            return False
        
        nodes = agent_data.get('nodes', {})
        connections = agent_data.get('connections', [])
        
        # Validar que todas las conexiones apunten a nodos existentes
        for conn in connections:
            src = conn.get('source_node')
            tgt = conn.get('target_node')
            
            if src not in nodes:
                print(f"❌ Nodo origen no existe: {src}")
                return False
            
            if tgt not in nodes:
                print(f"❌ Nodo destino no existe: {tgt}")
                return False
        
        print(f"✓ Estructura válida")
        print(f"✓ {len(nodes)} nodos")
        print(f"✓ {len(connections)} conexiones")
        print(f"✓ Todas las conexiones válidas")
        
        return True
    
    except Exception as e:
        print(f"❌ Error de validación: {str(e)}")
        return False


def list_all_workflows(projects_dir: str = "projects"):
    """Listar todos los workflows disponibles."""
    print(f"\n{'='*70}")
    print(f"Workflows Disponibles en {projects_dir}/")
    print(f"{'='*70}\n")
    
    projects_path = Path(projects_dir)
    buildbm_files = list(projects_path.glob("*.buildbm"))
    json_files = list(projects_path.glob("*.json"))
    
    all_files = buildbm_files + json_files
    
    if not all_files:
        print("No hay workflows guardados")
        return
    
    for file_path in sorted(all_files):
        size_kb = file_path.stat().st_size / 1024
        print(f"  📄 {file_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           BuildBrain - Gestor de Workflows                       ║
    ║                Ejemplo de Uso                                    ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Listar workflows disponibles
    list_all_workflows()
    
    # Cargar y mostrar el flujo principal
    workflow_file = "projects/Consulta_Usuarios_Deuda.buildbm"
    load_and_display_workflow(workflow_file)
    
    # Validar el flujo
    validate_workflow(workflow_file)
    
    # Cargar otro ejemplo
    print("\n" + "="*70)
    print("Cargando segundo ejemplo...")
    print("="*70)
    
    workflow_file2 = "projects/Analisis_Sentimiento.buildbm"
    load_and_display_workflow(workflow_file2)
    validate_workflow(workflow_file2)
    
    print("\n✓ Ejemplos completados")
    print("\n💡 Tip: Usa 'Load Agent' en la UI para cargar estos workflows")
    print("   y luego 'Execute' para ejecutarlos.\n")
