from src.gui.nodes_panel import CanvasView
from src.nodes.base import NodeCanvas
from src.nodes.types import NodeFactory
canvas = NodeCanvas()
canvas_view = CanvasView()
canvas_view.page = type("DummyPage", (), {"update": lambda self=None, *args, **kwargs: None, "window_width": 1400, "window_height": 900})()
canvas_view.set_canvas(canvas)
d1 = NodeFactory.create_node("dataset", x=50, y=50)
if d1:
    canvas.add_node(d1)
    canvas_view.add_visual(d1)
m1 = NodeFactory.create_node("model", x=250, y=100)
if m1:
    canvas.add_node(m1)
    canvas_view.add_visual(m1)
canvas_view._on_port_click(d1.node_id, "output")
canvas_view._on_port_click(m1.node_id, "input")
print('connections', len(canvas.connections))
print('stack controls', len(canvas_view.stack.controls))
print('first control', type(canvas_view.stack.controls[0]))
for conn_id, segments in canvas_view.connection_lines.items():
    print('conn', conn_id, len(segments))
    for seg in segments:
        print('seg', seg.left, seg.top, seg.width, seg.height)
