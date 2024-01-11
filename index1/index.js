function openModal() {
  document.getElementById("myModal").style.display = "block";
}

function closeModal() {
  console.log('closeModal() called');
  document.getElementById("myModal").style.display = "none";
}

window.addEventListener('click', function (event) {
  if (event.target == document.getElementById("myModal")) {
    closeModal();
  }
});

document.querySelector('.close').addEventListener('click', function () {
  closeModal();
});

function myFunction1() {
  var xhr = new XMLHttpRequest();
  var data = JSON.stringify({
    script_path: "E:/yolov5-5.0/detect.py"
  });
  xhr.open("POST", "http://127.0.0.1:5000/detect");
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  xhr.onreadystatechange = function () {
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
      var result = JSON.parse(this.responseText);
      document.getElementById("result").innerHTML = result;
    }
  }
  xhr.send(data);
}
function openNewPage() {
  window.open("test.html");
}
function openModal() {
  document.getElementById("myModal").style.display = "block";
}

function closeModal() {
  document.getElementById("myModal").style.display = "none";
}

function uploadFileAndRedirect() {
  var fileInput = document.getElementById("fileInput");
  if (fileInput.files.length > 0) {
    // Handle file upload and redirection here
    window.location.href = 'newPage.html'; // Replace with the URL of the new page
  } else {
    alert("请先选择一个文件");
  }
}

var video = document.querySelector('video');
var progressBar = document.querySelector('progress');

video.addEventListener('timeupdate', function() {
  var percentage = Math.floor((100 / video.duration) * video.currentTime);
  progressBar.value = percentage;
});

