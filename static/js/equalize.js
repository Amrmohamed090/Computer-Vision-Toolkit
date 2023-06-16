const input = document.querySelector("#image_input")
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")
let histogram = document.getElementById("histogram")
let equalize = document.getElementById("equalize")
let normalize = document.getElementById("normalize")
let local_thresholding = document.getElementById("local_thresholding")
let global_thresholding = document.getElementById("global_thresholding")
let rgb_histogram = document.getElementById("rgb_histogram")
let reset_btn = document.getElementById("reset_current_img")
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'
let sizethr = document.getElementById("sizethr")
let thr = document.getElementById("thr")
function load_img(src , id){
  let imgdiv = document.getElementById(id)
  
  let img = document.createElement('img');
  img.src = src;
  img.style.cssText=`
  height:350px;
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
        load_img("../static/img/input/uploaded.png", "img_org_div")
      }
  
  
    })
})
histogram.addEventListener("click", e=>{
  console.log("histogram")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "histogram",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }
  


  })
 
 
})
equalize.addEventListener("click", e=>{
  console.log("equalize")
  
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "equalize",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
normalize.addEventListener("click", e=>{
  console.log("normalize")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "normalize",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})

local_thresholding.addEventListener("click", e=>{
  console.log("local_thresholding")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "local_thresholding",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
})
global_thresholding.addEventListener("click", e=>{
  console.log("global_thresholding")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "global_thresholding",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
rgb_histogram.addEventListener("click", e=>{
  console.log("rgb_histogram")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/freq_request',
    data: {
      req_type: "rgb_histogram",
      thr:thr.value,
      sizethr:sizethr.value

    },
    success: function (res) {
      srcs = JSON.parse(res)["1"]
      let imgdiv = document.getElementById("img_res_div")
      let cont = document.createElement('div')
      cont.style.display = "inline-block"

      let img1 = document.createElement('img');
      img1.style.cssText=`     
      height: 400px;
      width: 300px;
      position: relative;
      top: -18px;
      left: -18px;
      `
      
      img1.src = srcs[0];
      let img2 = document.createElement('img');
      img2.style.cssText=`
      height: 400px;
      width: 300px;
      position: relative;
      top: -424px;
      left: 309px;
  
      `
      
      img2.src = srcs[1];

      // clean result before
      imgdiv.innerHTML = '';
      // append new image
      cont.appendChild(img1)
      cont.appendChild(img2)
      imgdiv.appendChild(cont)
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
                load_img(responce["1"] , "img_res_div")
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


load_img("./static/img/input/uploaded.png" , "img_org_div")
load_img("./static/img/input/current.png" , "img_res_div")