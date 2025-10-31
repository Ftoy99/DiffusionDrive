import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/Line2.js';
import { LineMaterial } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/LineMaterial.js';
import { LineGeometry } from 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/lines/LineGeometry.js';

async function initScene() {
    const container = document.getElementById('viewer');
    container.innerHTML = '';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);

    // Load scenario data
    const res = await fetch('/scenario_data');
    const data = await res.json();

    // Camera & renderer
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(20, 20, 20);
    camera.lookAt(scene.position);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio * 0.75);
    renderer.shadowMap.enabled = false;
    renderer.outputEncoding = THREE.LinearEncoding;
    container.appendChild(renderer.domElement);

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

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

    // Boxes using InstancedMesh
    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshBasicMaterial({ color: 0x0000ff, transparent: true, opacity: 0.3 });
    const boxMesh = new THREE.InstancedMesh(boxGeo, boxMat, data.bboxes.length);
    const dummy = new THREE.Object3D();

    data.bboxes.forEach((box, i) => {
        const [x, y, heading, length, width] = box;
        dummy.position.set(x, 0, y);
        dummy.scale.set(width, 1, length);
        dummy.rotation.y = -heading + Math.PI / 2;
        dummy.updateMatrix();
        boxMesh.setMatrixAt(i, dummy.matrix);
    });
    scene.add(boxMesh);


    // Camera window UI
    const cameraEl = document.getElementById('camera-window');
    cameraEl.src = data.image;

    // Camera window UI
    const gazeEl = document.getElementById('gaze-window');
    gazeEl.src = data.gaze_image;

    // Handle resize once
    function onResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        lineMaterials.forEach(m => m.resolution.set(renderer.domElement.width, renderer.domElement.height));
    }
    window.addEventListener('resize', onResize);
    onResize();

    // Animation loop
    function animationLoop() {
        controls.update();
        lineMaterials.forEach(m => m.resolution.set(window.innerWidth, window.innerHeight));
        renderer.render(scene, camera);
        requestAnimationFrame(animationLoop);
    }

    renderer.setAnimationLoop(animationLoop);
}

window.initScene = initScene;
