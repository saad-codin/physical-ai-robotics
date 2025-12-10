import React, { useEffect, useRef } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import * as THREE from 'three';
import styles from './styles.module.css';

function Robot3DComponent() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    renderer.setSize(width, height);
    containerRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x4299E1, 2, 100);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x667EEA, 2, 100);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);

    // Create robot
    const robotGroup = new THREE.Group();

    // Head
    const headGeometry = new THREE.BoxGeometry(1.2, 1, 1);
    const headMaterial = new THREE.MeshPhongMaterial({
      color: 0x2D3748,
      emissive: 0x4299E1,
      emissiveIntensity: 0.2,
      shininess: 100
    });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.5;
    robotGroup.add(head);

    // Eyes
    const eyeGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    const eyeMaterial = new THREE.MeshBasicMaterial({
      color: 0x4299E1,
      emissive: 0x4299E1,
      emissiveIntensity: 1
    });

    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.3, 1.6, 0.5);
    robotGroup.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.3, 1.6, 0.5);
    robotGroup.add(rightEye);

    // Antenna
    const antennaGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.5, 8);
    const antennaMaterial = new THREE.MeshPhongMaterial({
      color: 0x4299E1,
      emissive: 0x4299E1,
      emissiveIntensity: 0.5
    });
    const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
    antenna.position.y = 2.25;
    robotGroup.add(antenna);

    const antennaBallGeometry = new THREE.SphereGeometry(0.1, 16, 16);
    const antennaBall = new THREE.Mesh(antennaBallGeometry, eyeMaterial);
    antennaBall.position.y = 2.5;
    robotGroup.add(antennaBall);

    // Body
    const bodyGeometry = new THREE.BoxGeometry(1.5, 1.2, 0.8);
    const body = new THREE.Mesh(bodyGeometry, headMaterial);
    body.position.y = 0.3;
    robotGroup.add(body);

    // Chest panel
    const panelGeometry = new THREE.BoxGeometry(1, 0.8, 0.05);
    const panelMaterial = new THREE.MeshPhongMaterial({
      color: 0x1A202C,
      emissive: 0x4299E1,
      emissiveIntensity: 0.3
    });
    const panel = new THREE.Mesh(panelGeometry, panelMaterial);
    panel.position.set(0, 0.3, 0.43);
    robotGroup.add(panel);

    // Arms
    const armGeometry = new THREE.CylinderGeometry(0.15, 0.15, 1, 8);
    const leftArm = new THREE.Mesh(armGeometry, headMaterial);
    leftArm.position.set(-1, 0.3, 0);
    leftArm.rotation.z = Math.PI / 6;
    robotGroup.add(leftArm);

    const rightArm = new THREE.Mesh(armGeometry, headMaterial);
    rightArm.position.set(1, 0.3, 0);
    rightArm.rotation.z = -Math.PI / 6;
    robotGroup.add(rightArm);

    // Hands
    const handGeometry = new THREE.SphereGeometry(0.2, 16, 16);
    const leftHand = new THREE.Mesh(handGeometry, eyeMaterial);
    leftHand.position.set(-1.3, -0.2, 0);
    robotGroup.add(leftHand);

    const rightHand = new THREE.Mesh(handGeometry, eyeMaterial);
    rightHand.position.set(1.3, -0.2, 0);
    robotGroup.add(rightHand);

    // Add glow particles around robot
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 100;
    const posArray = new Float32Array(particlesCount * 3);

    for(let i = 0; i < particlesCount * 3; i++) {
      posArray[i] = (Math.random() - 0.5) * 5;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    const particlesMaterial = new THREE.PointsMaterial({
      size: 0.05,
      color: 0x4299E1,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });

    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    scene.add(robotGroup);
    camera.position.z = 5;

    // Mouse interaction
    let mouseX = 0;
    let mouseY = 0;

    const handleMouseMove = (event) => {
      const rect = containerRef.current.getBoundingClientRect();
      mouseX = ((event.clientX - rect.left) / width) * 2 - 1;
      mouseY = -((event.clientY - rect.top) / height) * 2 + 1;
    };

    window.addEventListener('mousemove', handleMouseMove);

    // Animation
    const animate = () => {
      requestAnimationFrame(animate);

      // Smooth robot rotation following mouse
      robotGroup.rotation.y += (mouseX * 0.5 - robotGroup.rotation.y) * 0.05;
      robotGroup.rotation.x += (mouseY * 0.3 - robotGroup.rotation.x) * 0.05;

      // Floating animation
      robotGroup.position.y = Math.sin(Date.now() * 0.001) * 0.2;

      // Rotate particles
      particlesMesh.rotation.y += 0.001;

      // Pulsing lights
      pointLight1.intensity = 2 + Math.sin(Date.now() * 0.002) * 0.5;
      pointLight2.intensity = 2 + Math.cos(Date.now() * 0.002) * 0.5;

      // Pulsing eyes
      const eyeIntensity = 1 + Math.sin(Date.now() * 0.003) * 0.3;
      leftEye.material.emissiveIntensity = eyeIntensity;
      rightEye.material.emissiveIntensity = eyeIntensity;

      renderer.render(scene, camera);
    };

    animate();

    // Cleanup
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div ref={containerRef} className={styles.robotContainer} />
  );
}

export default function Robot3D() {
  return (
    <BrowserOnly fallback={<div>Loading 3D Robot...</div>}>
      {() => <Robot3DComponent />}
    </BrowserOnly>
  );
}
