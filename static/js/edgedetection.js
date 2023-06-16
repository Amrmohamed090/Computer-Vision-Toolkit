const input = document.querySelector("#image_input")
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")
let sobel = document.getElementById("sobel")
let roberts = document.getElementById("roberts")
let prewitt = document.getElementById("prewitt")
let canny = document.getElementById("canny")
let reset_btn = document.getElementById("reset_current_img")
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'
let low_threshold = document.getElementById("low_threshold")
// let last_low_threshold_value = low_threshold.value
let high_threshold = document.getElementById("high_threshold")
// let last_high_threshold_value = high_threshold.value
let sigmaEd = document.getElementById("sigmaEd")
// low_threshold.addEventListener('input', e=>{
//   if (low_threshold.value < high_threshold.value && low_threshold.value <=0.98){
//     last_low_threshold_value = low_threshold.value
//   }
//   else{
//     low_threshold.value = last_low_threshold_value
//   }
// })

// high_threshold.addEventListener('input', e=>{
//   if (low_threshold.value < high_threshold.value && high_threshold.value <=0.99){
//     last_high_threshold_value = high_threshold.value
//   }
//   else{
//     high_threshold.value = last_high_threshold_value
//   }
// })

// let size = document.getElementById("sizeEd")
// let sigma = document.getElementById("sigma")
// let D0 = document.getElementById("D0")

function load_img(src , id){
  let imgdiv = document.getElementById(id)
  let img = document.createElement('img');
  img.src = src;
  img.style.cssText=`
  height:370px;

  `
  // clean result before
  imgdiv.innerHTML = '';
  // append new image
  imgdiv.appendChild(img)
}
reset_btn.addEventListener("click", e =>{
    e.preventDefault()
    $.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:5000/reset_current_img',
        
        success: function (res) {
          load_img(JSON.parse(res)["1"], "img_res_div")
        }
    
    
      })
})
sobel.addEventListener("click", e=>{
  console.log("sobel")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  // console.log(size.value)
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "sobel",
      sigma:1,
      size:3,
      D0:50

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})
roberts.addEventListener("click", e=>{
  console.log("roberts")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "roberts",
      sigma:1,
      size:3,
      D0:50

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
prewitt.addEventListener("click", e=>{
  console.log("prewitt")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "prewitt",
      sigma:1,
      size:3,
      D0:50

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
  
    }


  })
})

canny.addEventListener("click", e=>{
  console.log("canny")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "canny",
      sigma:1,
      size:3,
      D0:50,
      low_thresh:low_threshold.value,
      high_thresh:high_threshold.value,
      sigmaEd:sigmaEd.value
      

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})
//start upload button
input.addEventListener('change', e => {
  e.preventDefault()
  if (e.target.files.length) {
      // start file reader
      const reader = new FileReader();
      reader.onload = e => {
        
        if (e.target.result) {
   
          load_img(e.target.result, "img_org_div")
          original_img_1 = e.target.result
          $.ajax({
            type: 'POST',
            url: 'http://127.0.0.1:5000/saveImg',
            data: {
              img_upload: original_img_1,
        
            },
            success: function (res) {
                var responce = JSON.parse(res)
                console.log(responce)
      
            }
      
      
          })
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });
//end upload button

//getting the uploaded image from the backend so it won't be lost if we go from one directory to another
//the photo is saved in a session state in the backend 

//let current_img_src = document.getElementById("current_img_source").innerHTML

load_img("./static/img/input/current.png" , "img_org_div")
