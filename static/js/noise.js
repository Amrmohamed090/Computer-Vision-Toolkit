const input = document.querySelector("#image_input")
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")
let uniform_noise = document.getElementById("uniform_noise")
let gaussian_noise = document.getElementById("gaussian_noise")
let salt_noise = document.getElementById("salt_noise")
let average_filter = document.getElementById("average_filter")
let median_filter = document.getElementById("median_filter")
let gaussian_filter = document.getElementById("gaussian_filter")
let low_pass = document.getElementById("low_pass")
let high_pass = document.getElementById("high_pass")
let reset_btn = document.getElementById("reset_current_img")
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'
let size = document.getElementById("size")
let sigma = document.getElementById("sigma")
let D0 = document.getElementById("D0")
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
uniform_noise.addEventListener("click", e=>{
 load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("uniform_noise")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "uniform_noise",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
gaussian_noise.addEventListener("click", e=>{
  console.log("gaussian_noise")
  
  load_img(img_res_div.childNodes[0].src , "img_org_div")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "gaussian_noise",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
salt_noise.addEventListener("click", e=>{
  console.log("salt_noise")
  load_img(img_res_div.childNodes[0].src , "img_org_div")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "salt_noise",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})

average_filter.addEventListener("click", e=>{
  console.log("average_filter")
  load_img(img_res_div.childNodes[0].src , "img_org_div")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "average_filter",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
})
median_filter.addEventListener("click", e=>{
  console.log("median_filter")
  load_img(img_res_div.childNodes[0].src , "img_org_div")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "median_filter",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
  
})
gaussian_filter.addEventListener("click", e=>{
  console.log("gaussian_filter")
 
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "gaussian_filter",
      size: size.value,
      sigma:sigma.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
})
low_pass.addEventListener("click", e=>{
  console.log("low_pass")
  
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "low_pass",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
      
    }


  })
})
high_pass.addEventListener("click", e=>{
  console.log("high_pass")
  load_img(img_res_div.childNodes[0].src , "img_org_div")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/noise_req',
    data: {
      noise_type: "high_pass",
      sigma:sigma.value,
      size:size.value,
      D0:D0.value

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
