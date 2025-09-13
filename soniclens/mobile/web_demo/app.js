const video = document.getElementById('video');
const overlay = document.getElementById('overlay');

async function startCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:false });
  video.srcObject = stream;
}
startCamera();

// simple polling demo: fetch latest caption from server every 1s
async function pollCaption(){
  try {
    const res = await fetch('http://localhost:8000/latest_caption');
    if(res.ok){
      const j = await res.json();
      overlay.innerText = j.caption || "No caption yet";
    }
  } catch(e) {
    overlay.innerText = "Backend not running";
  }
}
setInterval(pollCaption, 1000);
