// Real robot and camera, for use in the lab on the robots' Wi-Fi.
//
// Shadow: the page connects to a real Niryo One's rosbridge (ws://<robot IP>:9090), the same connection the
// ROS exercise's shadow.py uses, and the 3D arm follows the real arm's joints. Nothing is sent to the robot
// except subscriptions: the robot is never commanded from here.
//
// Camera: a USB camera (for example the station camera plugged into a laptop) replaces the simulated camera
// for the programs and the preview. Settings the programs make (focus, exposure, white balance) are passed
// to the camera where the browser allows it.
import * as S from './scene.js';

const $ = id => document.getElementById(id);
const say = (msg, cls) => { const el = $('realstatus'); el.textContent = msg; el.className = cls || ''; };

// ---------------------------------------------------------------- shadow (rosbridge, read only)
let ws = null, lastMsg = 0, watchdog = null, hw = {}, learning = null;
const ARM_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'];

function subscribe(topic, type, throttle) {
  ws.send(JSON.stringify({ op: 'subscribe', id: 'sim_' + topic, topic, type, throttle_rate: throttle || 0, queue_length: 1 }));
}

function describe() {
  const parts = [];
  if (hw.calibration_needed) parts.push('NOT CALIBRATED: the 3D arm will not match until the robot is calibrated (ask the technician)');
  else if (hw.calibration_in_progress) parts.push('calibrating...');
  if (learning !== null) parts.push(learning ? 'learning mode ON' : 'learning mode off');
  return parts.join(' · ');
}

export function startShadow() {
  const ip = $('realip').value.trim();
  if (!/^\d{1,3}(\.\d{1,3}){3}$/.test(ip)) { say('Type the robot\'s IP address, e.g. 192.168.1.102 (on the label of your platform).', 'bad'); return; }
  stopShadow(true);
  say(`Connecting to ${ip}:9090... (Chrome or Edge may ask to "connect to devices on your local network": press Allow)`);
  try {
    ws = new WebSocket(`ws://${ip}:9090`);     // started by a click, so the browser can ask for permission
  } catch (e) {
    say('The browser refused the connection: ' + e.message + '. Use Chrome or Edge (recent), or open the page with serve.py on this PC.', 'bad');
    ws = null; return;
  }
  const noAnswer = `No answer from ${ip}:9090. Check: this computer is on the robots' Wi-Fi, the robot is on, the IP is right, and "local network" access is allowed for this site (padlock icon > Site settings).`;
  const timer = setTimeout(() => {
    if (ws && ws.readyState !== WebSocket.OPEN) { stopShadow(true); say(noAnswer, 'bad'); }
  }, 6000);
  ws.onopen = () => {
    clearTimeout(timer);
    subscribe('/joint_states', 'sensor_msgs/JointState', 40);
    subscribe('/niryo_one/hardware_status', 'niryo_one_msgs/HardwareStatus', 1000);
    subscribe('/niryo_one/learning_mode', 'std_msgs/Bool', 200);
    S.setShadow(true);
    lastMsg = performance.now();
    $('shadowbtn').textContent = 'Stop shadow';
    say(`Connected to ${ip}. Waiting for the robot's joints...`, 'ok');
    watchdog = setInterval(() => {
      if (performance.now() - lastMsg > 3000) say(`Connected to ${ip}, but no joint positions for 3 s: is the robot's ROS running?`, 'bad');
    }, 1000);
  };
  ws.onmessage = ev => {
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.op !== 'publish') return;
    if (m.topic === '/joint_states') {
      const names = m.msg.name || [], pos = m.msg.position || [];
      const j = ARM_JOINTS.map((n, i) => { const k = names.indexOf(n); return k >= 0 ? pos[k] : pos[i]; });
      if (j.some(v => typeof v !== 'number' || !isFinite(v))) return;
      S.shadowJoints(j);
      lastMsg = performance.now();
      say(`SHADOW of ${ip}: ` + (describe() || 'following the real arm') + `   [${j.map(v => v.toFixed(2)).join(', ')}]`, hw.calibration_needed ? 'bad' : 'ok');
    } else if (m.topic === '/niryo_one/hardware_status') {
      hw = m.msg || {};
    } else if (m.topic === '/niryo_one/learning_mode') {
      learning = !!(m.msg && m.msg.data);
    }
  };
  ws.onerror = () => {};
  ws.onclose = () => {
    clearTimeout(timer);
    if (ws) { const was = S.robot.shadow; stopShadow(true); say(was ? `Connection to ${ip} closed: the 3D arm is the simulated robot again.` : noAnswer, 'bad'); }
  };
}

export function stopShadow(quiet) {
  if (watchdog) { clearInterval(watchdog); watchdog = null; }
  const w = ws; ws = null;
  if (w) { try { w.onclose = null; w.close(); } catch (e) {} }
  if (S.robot.shadow) S.setShadow(false);
  hw = {}; learning = null;
  $('shadowbtn').textContent = 'Start shadow';
  if (!quiet) say('Shadow stopped. The 3D arm is the simulated robot again.');
}

// ---------------------------------------------------------------- USB camera
let stream = null, video = null, canvas = null, ctx = null;
const caps = () => { try { return stream.getVideoTracks()[0].getCapabilities(); } catch (e) { return {}; } };
const track = () => stream && stream.getVideoTracks()[0];

export async function listCameras() {
  const sel = $('camsel');
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const keep = sel.value;
    sel.innerHTML = '<option value="sim">Simulated station camera</option>' +
      devs.map((d, i) => `<option value="${d.deviceId}">${d.label || 'USB camera ' + (i + 1)}</option>`).join('');
    if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
  } catch (e) { /* no camera access */ }
}

export async function useCamera(id) {
  stopCamera();
  if (id === 'sim') { say('Simulated station camera.'); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: {
      deviceId: id === 'any' ? undefined : { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    say('Camera not available: ' + e.message + ' (allow the camera for this site).', 'bad'); $('camsel').value = 'sim'; return;
  }
  video = document.createElement('video');
  video.muted = true; video.playsInline = true; video.srcObject = stream;
  await video.play().catch(() => {});
  canvas = document.createElement('canvas'); ctx = canvas.getContext('2d', { willReadFrequently: true });
  S.setExternalCamera(realFrame);
  const s = track().getSettings();
  say(`USB camera in use: ${track().label || 'camera'} (${s.width} x ${s.height}). The programs and the preview now use it.`, 'ok');
  listCameras();
}

export function stopCamera() {
  S.setExternalCamera(null);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video = null;
}

// Frames in the same layout as the simulated camera's (RGBA, bottom row first).
function realFrame(w, h) {
  if (!video || video.readyState < 2) return null;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.save(); ctx.translate(0, h); ctx.scale(1, -1); ctx.drawImage(video, 0, 0, w, h); ctx.restore();
  return new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
}

export function cameraIsReal() { return !!stream; }

// Settings from the programs (cv2 property numbers, as in OpenCV).
const P = { AUTO_EXPOSURE: 21, EXPOSURE: 15, AUTO_WB: 44, WB_TEMPERATURE: 45, AUTOFOCUS: 39, FOCUS: 28, BRIGHTNESS: 10, CONTRAST: 11, SATURATION: 12 };
const scale = (v, lo, hi, range) => range ? range.min + (range.max - range.min) * Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : undefined;

export async function cameraSet(prop, value) {
  const t = track(); if (!t) return { ok: false, applied: false };
  const c = caps(), adv = {};
  if (prop === P.AUTOFOCUS && c.focusMode) adv.focusMode = value >= 0.5 ? 'continuous' : 'manual';
  if (prop === P.FOCUS && c.focusDistance) { adv.focusMode = 'manual'; adv.focusDistance = scale(value, 0, 255, c.focusDistance); }
  if (prop === P.AUTO_EXPOSURE && c.exposureMode) adv.exposureMode = (value >= 2 || value === 0.75) ? 'continuous' : 'manual';
  if (prop === P.EXPOSURE && c.exposureTime) {
    adv.exposureMode = 'manual';
    // OpenCV on Windows uses log2(seconds) (e.g. -6); elsewhere a time in 100 us units.
    const t100us = value <= 0 ? Math.pow(2, value) * 10000 : value;
    adv.exposureTime = Math.min(c.exposureTime.max, Math.max(c.exposureTime.min, t100us));
  }
  if (prop === P.AUTO_WB && c.whiteBalanceMode) adv.whiteBalanceMode = value >= 0.5 ? 'continuous' : 'manual';
  if (prop === P.WB_TEMPERATURE && c.colorTemperature) {
    adv.whiteBalanceMode = 'manual';
    adv.colorTemperature = Math.min(c.colorTemperature.max, Math.max(c.colorTemperature.min, value));
  }
  if (prop === P.BRIGHTNESS && c.brightness) adv.brightness = scale(value, 0, 255, c.brightness);
  if (prop === P.CONTRAST && c.contrast) adv.contrast = scale(value, 0, 255, c.contrast);
  if (prop === P.SATURATION && c.saturation) adv.saturation = scale(value, 0, 255, c.saturation);
  if (!Object.keys(adv).length) return { ok: true, applied: false };
  try { await t.applyConstraints({ advanced: [adv] }); return { ok: true, applied: true }; }
  catch (e) { return { ok: true, applied: false, error: String(e) }; }
}

export function cameraGet(prop) {
  const t = track(); if (!t) return { value: null };
  const s = t.getSettings();
  if (prop === P.WB_TEMPERATURE && s.colorTemperature) return { value: s.colorTemperature };
  if (prop === P.FOCUS && s.focusDistance !== undefined && caps().focusDistance) {
    const r = caps().focusDistance; return { value: 255 * (s.focusDistance - r.min) / ((r.max - r.min) || 1) };
  }
  return { value: null };
}

// ---------------------------------------------------------------- page controls
export function initReal() {
  $('shadowbtn').onclick = () => (S.robot.shadow || ws) ? stopShadow() : startShadow();
  $('camsel').onchange = () => useCamera($('camsel').value);
  $('camsel').onfocus = () => listCameras();
  $('camfind').onclick = async () => {
    // Labels and device ids are only shown after the camera has been allowed once.
    try { const s = await navigator.mediaDevices.getUserMedia({ video: true }); s.getTracks().forEach(t => t.stop()); } catch (e) {
      say('Camera not allowed: ' + e.message, 'bad'); return;
    }
    await listCameras();
    if (!stream) say('Cameras found: choose one in the Camera list.');
  };
  try { const saved = localStorage.getItem('sim_real_ip'); if (saved) $('realip').value = saved; } catch (e) {}
  $('realip').onchange = () => { try { localStorage.setItem('sim_real_ip', $('realip').value.trim()); } catch (e) {} };
  listCameras();
}
