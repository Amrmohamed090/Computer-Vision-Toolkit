const input = document.querySelector("#image_input")
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")

let kmean = document.getElementById("kmean")
let k = document.getElementById("k")
let max_iteration = document.getElementById("max_iteration")

let region_growing = document.getElementById("region-growing")
let seed_x = document.getElementById("seed-x")
let seed_y = document.getElementById("seed-y")
let thresh_region = document.getElementById("thresh_region")

let agglomerative = document.getElementById("agglomerative")
let number_of_clusters = document.getElementById("number_of_clusters")

let mean_shift = document.getElementById("mean-shift")
let raduis = document.getElementById("raduis")
let threshold_mean = document.getElementById("threshold_mean")

let color_space = document.getElementById("color_space")
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'







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

  color_space.addEventListener("click", e=>{
    if (color_space.value == "LUV"){
      $.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:5000/rgb_to_luv',
        data: {
    
        },
        success: function (res) {
          load_img(JSON.parse(res)["1"], "img_org_div")
         
        }
    
    
      })
    }
    else if (color_space.value == "RGB"){
      load_img("./static/img/input/current.png", "img_org_div")
    }
  })
  kmean.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("kmean")
  e.preventDefault()
  img_res_div.innerHTML = buffering_logo
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/segmentation_req',
    data: {
      segmentation_type: "kmean",
      k: k.value,
      max_iteration:max_iteration.value,
      color_space:color_space.value

    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})
region_growing.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("region_growing")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/segmentation_req',
    data: {
        segmentation_type: "region_growing",
      seed_x: seed_x.value,
      seed_y:seed_y.value,
      thresh_region:thresh_region.value


    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})

agglomerative.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("agglomerative")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/segmentation_req',
    data: {
      segmentation_type: "agglomerative",
      number_of_clusters:number_of_clusters.value,
      color_space:color_space.value


    },
    success: function (res) {
      load_img(JSON.parse(res)["1"], "img_res_div")
     
    }


  })
})
mean_shift.addEventListener("click", e=>{
  // load_img(img_res_div.childNodes[0].src , "img_org_div")
  console.log("mean_shift")
  img_res_div.innerHTML = buffering_logo
  e.preventDefault()
  $.ajax({
    type: 'POST',
    url: 'http://127.0.0.1:5000/segmentation_req',
    data: {
        segmentation_type: "mean_shift",
      raduis: raduis.value,
      threshold_mean:threshold_mean.value,
      color_space:color_space.value


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
              luv:true
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