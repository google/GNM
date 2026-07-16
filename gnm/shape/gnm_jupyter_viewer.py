"""GNM Jupyter viewer."""

import json
import uuid

from gnm.shape.visualization import vertex_colors as vertex_colors_module
from IPython.display import HTML, Javascript, display
import ipywidgets as widgets
import numpy as np

_get_vertex_colors = vertex_colors_module.get_vertex_colors


class GNMMeshViewer:
  """Sets up a three.js viewer for G-Nome meshes in Jupyter Notebook."""

  def __init__(
      self,
      gnm,
      width=480,
      height=640,
      fov=15,
      with_shadow=True,
      with_wireframe=False,
      parent_dom_id=None,
  ):
    self.width = width
    self.height = height
    self.vertices = gnm.template_vertex_positions
    self.gnm = gnm

    # Generate a unique ID for this viewer instance
    self.viewer_id = f"gnm_viewer_{uuid.uuid4().hex}"
    self.out = widgets.Output(
        layout=widgets.Layout(height="0", margin="0", padding="0")
    )

    # Prepare geometry data
    self.triangles = self.gnm.triangles_group("~eye_exteriors")
    self.vertex_colors = _get_vertex_colors(gnm_np=self.gnm)
    if "pupils" in self.gnm.vertex_group_names:
      self.vertex_colors[self.gnm.vertex_group_indices("pupils")] = 0.0

    self._init_html(
        width, height, fov, with_shadow, with_wireframe, parent_dom_id
    )

  def _init_html(
      self, width, height, fov, with_shadow, with_wireframe, parent_dom_id
  ):
    container_id = f"container_{self.viewer_id}"

    html_content = ""
    if parent_dom_id is None:
      html_content = (
          f'<div id="{container_id}" style="width: {width}px; height:'
          f' {height}px;"></div>'
      )
      target_dom_id = container_id
    else:
      target_dom_id = parent_dom_id

    display(HTML(html_content))
    display(self.out)

    js_data = {
        "vertices": self.vertices.ravel().tolist(),
        "faces": self.triangles.ravel().tolist(),
        "colors": self.vertex_colors.ravel().tolist(),
        "width": width,
        "height": height,
        "fov": fov,
        "withShadow": with_shadow,
        "withWireframe": with_wireframe,
        "targetDomId": target_dom_id,
        "viewerId": self.viewer_id,
        "y_mean": float(self.vertices.mean(axis=0)[1]),
    }

    js_code = f"""
    (function() {{
      const data = {json.dumps(js_data)};

      const threeUrl = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r98/three.min.js";
      const controlsUrl = "https://cdn.jsdelivr.net/npm/three@0.98.0/examples/js/controls/OrbitControls.js";

      function loadScript(url) {{
        return new Promise((resolve, reject) => {{
          if (document.querySelector(`script[src="${{url}}"]`)) {{
            resolve();
            return;
          }}
          const script = document.createElement('script');
          script.src = url;
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        }});
      }}

      function init() {{
        const width = data.width;
        const height = data.height;

        const container = document.getElementById(data.targetDomId);
        if (!container) {{
          console.error("Container not found:", data.targetDomId);
          return;
        }}

        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(width, height);
        renderer.setClearColor(0xEFEFEF);
        renderer.clear();
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.shadowMapEnabled = true;
        container.appendChild(renderer.domElement);

        const scene = new THREE.Scene();

        const camera = new THREE.PerspectiveCamera(data.fov, width / height, 0.01, 100.0);
        camera.position.set(0.0, data.y_mean, 2.0);
        scene.add(camera);
        scene.fog = new THREE.Fog(0xEFEFEF, 3.0, 8.0);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.screenSpacePanning = true;
        controls.enableZoom = true;
        controls.target.set(0.0, data.y_mean, 0.0);
        controls.update();

        const light_1 = new THREE.DirectionalLight(0xEEEEFF, 1.1);
        light_1.position.set(1, 0.5, 2);
        const light_2 = new THREE.DirectionalLight(0xFFEEEE, 0.5);
        light_2.position.set(-20.0, 0, -40.0);
        [light_1, light_2].forEach(light => {{
          light.castShadow = true;
          light.shadow.camera.left = light.shadow.camera.bottom = -0.5;
          light.shadow.camera.right = light.shadow.camera.top = 0.5;
          light.shadowMapWidth = light.shadowMapHeight = 2048;
          camera.add(light);
        }});

        const geometry = new THREE.BufferGeometry();
        const verticesVal = new Float32Array(data.vertices);
        const facesVal = new Uint32Array(data.faces);
        const colorsVal = new Float32Array(data.colors);

        geometry.addAttribute(
            'position', new THREE.BufferAttribute(verticesVal, 3)
        );
        geometry.setIndex(new THREE.BufferAttribute(facesVal, 1));
        geometry.addAttribute('color', new THREE.BufferAttribute(colorsVal, 3));
        geometry.computeVertexNormals();

        const face_material = new THREE.MeshStandardMaterial({{
            metalness: 0.0,
            roughness: 0.6,
            vertexColors: THREE.VertexColors,
        }});

        const mesh = new THREE.Mesh(geometry, face_material);
        mesh.castShadow = data.withShadow;
        mesh.receiveShadow = data.withShadow;
        scene.add(mesh);

        if (data.withWireframe) {{
          const wireframe_material = new THREE.MeshBasicMaterial({{
              wireframe: true,
              color: 0x050505,
              polygonOffset: true,
              polygonOffsetFactor: -2,
              polygonOffsetUnits: -4,
          }});
          const wireframe_mesh = new THREE.Mesh(geometry, wireframe_material);
          scene.add(wireframe_mesh);
        }}

        const grid_helper = new THREE.GridHelper(10, 100, 0xAAAAAA, 0xDDDDDD);
        scene.add(grid_helper);
        const axis_helper = new THREE.AxesHelper(0.05);
        axis_helper.position.y = 1e-4;
        scene.add(axis_helper);

        const animate = function () {{
          requestAnimationFrame(animate);
          renderer.render(scene, camera);
        }};
        animate();

        window.gnmViewers = window.gnmViewers || {{}};
        window.gnmViewers[data.viewerId] = {{
          geometry: geometry,
          positionAttribute: geometry.getAttribute('position'),
          scene: scene,
          camera: camera,
          renderer: renderer
        }};
      }}

      loadScript(threeUrl)
        .then(() => loadScript(controlsUrl))
        .then(init)
        .catch(err => console.error("Failed to load scripts", err));
    }})();
    """
    display(Javascript(js_code))

  def update(self, vertices: np.ndarray) -> None:
    """Update G-Nome mesh vertices."""
    self.vertices = vertices
    vertices_list = vertices.ravel().tolist()
    js_code = f"""
    (function() {{
      const viewer = window.gnmViewers && window.gnmViewers['{self.viewer_id}'];
      if (viewer) {{
        const posAttr = viewer.positionAttribute;
        const newVertices = new Float32Array({vertices_list});
        posAttr.copyArray(newVertices);
        posAttr.needsUpdate = true;
        viewer.geometry.computeVertexNormals();
      }} else {{
        console.warn("Viewer not found for update:", '{self.viewer_id}');
      }}
    }})();
    """
    with self.out:
      display(Javascript(js_code))
