import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/LineGeometry.js';

// 🔹 declare globals safely
let renderer = null;
let resizeHandler = null;
let animationId = null;

async function initScene() {
    const container = document.getElementById('viewer');
    container.innerHTML = '';

    // 🔹 cleanup old context safely
    if (animationId) cancelAnimationFrame(animationId);
    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler);
        resizeHandler = null;
    }
    if (renderer) {
        renderer.dispose?.();
        renderer.forceContextLoss?.();
        renderer.domElement.remove();
        renderer = null;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);

    // Load scenario data
    const res = await fetch('/scenario_data');
    const data = await res.json();

    // Camera & renderer
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(-15, 15, 15);
    camera.lookAt(scene.position);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio * 0.75);
    renderer.shadowMap.enabled = false;
    renderer.outputEncoding = THREE.LinearEncoding;
    container.appendChild(renderer.domElement);

    // 🔹 Resize handler (remove old one first)
    resizeHandler = () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', resizeHandler);
    resizeHandler();

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.target.set(0, 0, 0);
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.ROTATE};
    controls.update();


    // LiDAR point cloud
    const N = data.lidar_raw.length;
    const positions = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
        const [x, y, z] = data.lidar_raw[i];
        positions[i * 3 + 0] = x;
        positions[i * 3 + 1] = z;
        positions[i * 3 + 2] = y;
    }
    const lidarGeometry = new THREE.BufferGeometry();
    lidarGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const lidarMaterial = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.03,
        sizeAttenuation: true,
        transparent: true,
        depthWrite: false
    });
    const lidarCloud = new THREE.Points(lidarGeometry, lidarMaterial);
    scene.add(lidarCloud);

    // Axis helper
    scene.add(new THREE.AxesHelper(50));
    // Line materials
    const lineMaterials = [];
    function makeLine(points, color, height = 0, width = 0.2) {
      const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], height, p[1])));
      const geometry = new THREE.TubeGeometry(curve, 64, width / 2, 8, false);
      const material = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);
    }

    data.trajectories.forEach(traj => makeLine(traj, 0x00ff00, 1));
    makeLine(data.true_trajectory, 0xff69b4, 2);
    makeLine(data.ego_trajectory, 0x0000ff, 0);
    makeLine(data.ego_trajectory_no_unreliables, 0x00FFFF, 1);

    // Boxes using InstancedMesh
    const vehicle_boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const vehicle_boxMat = new THREE.MeshBasicMaterial({ color: 0xff69b4, transparent: true, opacity: 0.3 });
    const vehicle_boxMesh = new THREE.InstancedMesh(vehicle_boxGeo, vehicle_boxMat, data.vehicle_bboxes.length);
    const vehicle_dummy = new THREE.Object3D();

    data.vehicle_bboxes.forEach((box, i) => {
        const [x, y, heading, length, width] = box;
        vehicle_dummy.position.set(x, 1, y);
        vehicle_dummy.scale.set(width, 1, length);
        vehicle_dummy.rotation.y = -heading + Math.PI / 2;
        vehicle_dummy.updateMatrix();
        vehicle_boxMesh.setMatrixAt(i, vehicle_dummy.matrix);
    });
    scene.add(vehicle_boxMesh);

    // Boxes using InstancedMesh
    const pedestrian_boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const pedestrian_boxMat = new THREE.MeshBasicMaterial({ color: 0xff69b4, transparent: true, opacity: 0.3 });
    const pedestrian_boxMesh = new THREE.InstancedMesh(pedestrian_boxGeo, pedestrian_boxMat, data.pedestrian_bboxes.length);
    const pedestrian_dummy = new THREE.Object3D();

    data.pedestrian_bboxes.forEach((box, i) => {
        const [x, y, heading, length, width] = box;
        pedestrian_dummy.position.set(x, 1, y);
        pedestrian_dummy.scale.set(width, 1, length);
        pedestrian_dummy.rotation.y = -heading + Math.PI / 2;
        pedestrian_dummy.updateMatrix();
        pedestrian_boxMesh.setMatrixAt(i, pedestrian_dummy.matrix);
    });
    scene.add(pedestrian_boxMesh);

    // Pred Boxes using InstancedMesh
    const pred_vehicle_boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const pred_vehicle_boxMat = new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.3 });
    const pred_vehicle_boxMesh = new THREE.InstancedMesh(pred_vehicle_boxGeo, pred_vehicle_boxMat, data.pred_vehicle_bboxes.length);
    const pred_vehicle_dummy = new THREE.Object3D();

    data.pred_vehicle_bboxes.forEach((box, i) => {
        const [x, y, heading, length, width] = box;
        pred_vehicle_dummy.position.set(x, 0, y);
        pred_vehicle_dummy.scale.set(width, 1, length);
        pred_vehicle_dummy.rotation.y = -heading + Math.PI / 2;
        pred_vehicle_dummy.updateMatrix();
        pred_vehicle_boxMesh.setMatrixAt(i, pred_vehicle_dummy.matrix);
    });
    scene.add(pred_vehicle_boxMesh);

    // Pred Boxes using InstancedMesh
    const pred_pedestrian_boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const pred_pedestrian_boxMat = new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.3 });
    const pred_pedestrian_boxMesh = new THREE.InstancedMesh(pred_pedestrian_boxGeo, pred_pedestrian_boxMat, data.pred_pedestrian_bboxes.length);
    const pred_pedestrian_dummy = new THREE.Object3D();

    data.pred_pedestrian_bboxes.forEach((box, i) => {
        const [x, y, heading, length, width] = box;
        pred_pedestrian_dummy.position.set(x, 0, y);
        pred_pedestrian_dummy.scale.set(width, 1, length);
        pred_pedestrian_dummy.rotation.y = -heading + Math.PI / 2;
        pred_pedestrian_dummy.updateMatrix();
        pred_pedestrian_boxMesh.setMatrixAt(i, pred_pedestrian_dummy.matrix);
    });
    scene.add(pred_pedestrian_boxMesh);

    // Draw stop line if it exists
    if (data.stop_line && data.stop_line.length > 0) {
        console.log('Stop line points:', data.stop_line);

        // Convert to Vector3 (Y = 0.5 to lift above ground)
        const curvePoints = data.stop_line.map(p => new THREE.Vector3(p[0], 0.5, p[1]));

        console.log(`Drawing stop line with ${curvePoints.length} points`);

        // Create a CatmullRomCurve3 for smoothness
        const curve = new THREE.CatmullRomCurve3(curvePoints);

        // Thick 3D tube
        const tubeGeometry = new THREE.TubeGeometry(curve, 64, 0.1, 8, false);
        const tubeMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 }); // yellow
        const tubeMesh = new THREE.Mesh(tubeGeometry, tubeMaterial);
        scene.add(tubeMesh);
    }

    // Camera window UI
    const cameraEl = document.getElementById('camera-window');
    cameraEl.src = data.image;

    // Camera window UI
    const gazeEl = document.getElementById('gaze-window');
    gazeEl.src = data.gaze_image;

    // Semantic window UI
    const semanticEl = document.getElementById('semantic-window');
    semanticEl.src = data.semantic;

    // Pred Semantic window UI
    const predSemanticEl = document.getElementById('pred_semantic-window');
    predSemanticEl.src = data.pred_semantic;

    // FPS / ms
    const fpsEl = document.getElementById('fps-counter');
    if (data.fps !== undefined && data.ms !== undefined) {
        fpsEl.textContent = `FPS: ${data.fps.toFixed(1)} | ms: ${data.ms.toFixed(1)}`;
    }

    // FPS / ms
    const predEl = document.getElementById('pred-counter');
    if (data.fps !== undefined && data.ms !== undefined) {
        predEl.textContent = ` LIGHT: ${data.light} | PRED LIGHT: ${data.pred_light}`;
    }

    // Handle resize once
    function onResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        lineMaterials.forEach(m => m.resolution.set(renderer.domElement.width, renderer.domElement.height));
    }
    window.addEventListener('resize', onResize);
    onResize();

    //  Animation loop (store frame ID for cancelation)
    function animationLoop() {
        controls.update();
        renderer.render(scene, camera);
        animationId = requestAnimationFrame(animationLoop);
    }
    animationLoop();
}

window.initScene = initScene;
