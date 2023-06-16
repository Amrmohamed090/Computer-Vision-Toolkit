const input = document.querySelector("#image_input")
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")
let houghLine = document.getElementById("houghLine")
let houghCircle = document.getElementById("houghCircle")
let reset_btn = document.getElementById("reset_current_img")
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'

let threshold = document.getElementById("thresholdHough")
let thetastep = document.getElementById("thetastep")
let rstep = document.getElementById("rstep")


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
        }
    
    
      })
})
houghLine.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("houghLine")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/hough_req',
    data: {
      hough_type: "houghLine",
      threshold:threshold.value,
      rstep:rstep.value,
      thetastep:thetastep.value,


    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})
houghCircle.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("houghCircle")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/hough_req',
    data: {
      hough_type: "houghCircle",
      threshold:threshold.value,
      rstep:rstep.value,
      thetastep:thetastep.value,


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