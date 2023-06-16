
let img_org_div = document.getElementById("img_org_div")
let img_res_div = document.getElementById("img_res_div")
const image_input1 = document.getElementById("image_input1")
const image_input1_html = ' <p class="uppercase text-xs text-gray-600 mb-2 tracking-wider ">Upload first Image</p><div class="Neon Neon-theme-dragdropbox "><input id="image_input1" type="file"><output></output><div class="Neon-input-dragDrop"><div class="Neon-input-inner"><div class="Neon-input-icon"><i class="fa fa-file-image-o"></i></div><div class="Neon-input-text"><h3>Drag&amp;Drop files here</h3> <span style="display:inline-block; margin: 15px 0">or</span></div><a class="Neon-input-choose-btn blue">Browse Files</a></div></div></div>'
const image_input2 = document.getElementById("image_input2")
const image_input2_html = '<p class="uppercase text-xs text-gray-600 mb-2 tracking-wider">Upload second Image</p><div class="Neon Neon-theme-dragdropbox"><input id="image_input2" type="file"><output></output><div class="Neon-input-dragDrop"><div class="Neon-input-inner"><div class="Neon-input-icon"><i class="fa fa-file-image-o"></i></div><div class="Neon-input-text"><h3>Drag&amp;Drop files here</h3> <span style="display:inline-block; margin: 15px 0">or</span></div><a class="Neon-input-choose-btn blue">Browse Files</a></div></div></div> '
let buffering_logo = '<img src="../static/img/loading.gif" id="buffering_logo" style="width: 10%; margin-top:25%; margin-left:25%;">'
let original_img_1 = ""
let original_img_2 = ""
let reset_btn = document.getElementById("reset_current_img")
let D0Low = document.getElementById("D0Low")
let D0High = document.getElementById("D0High")


reset_btn.addEventListener("click", e =>{
  document.getElementById("first_image").innerHTML=image_input1_html
  document.getElementById("second_image").innerHTML=image_input2_html
  document.getElementById("result1").innerHTML = ""
  document.getElementById("result2").innerHTML = ""
  let image_input11 = document.getElementById("image_input1")
  image_input11.addEventListener('change', e => {
    e.preventDefault()
    if (e.target.files.length) {
        // start file reader
        const reader = new FileReader();
        reader.onload = e => {
          
          if (e.target.result) {
     
            load_img(e.target.result, "first_image")
            original_img_1 = e.target.result
            if (document.getElementById("second_image").childNodes[0].nodeName == "IMG"){
  
                
                $.ajax({
                  type: 'POST',
                  url: 'http://127.0.0.1:5000/hybrid_req',
                  data: {
                    img_upload1: original_img_1,
                    img_upload2: original_img_2,
                    D0Low:D0Low.value,
                    D0High:D0High.value
                    
              
                  },
                  success: function (res) {
                      var responce = JSON.parse(res)
                      console.log(responce)
                      load_img(responce["1"],'result1')
                      load_img(responce["2"],'result2')
                    
                  }
            
            
                })
          }
          }
        };
        reader.readAsDataURL(e.target.files[0]);
      }
    });
    let image_input22 = document.getElementById("image_input2")
    image_input22.addEventListener('change', e => {
      e.preventDefault()
      if (e.target.files.length) {
          // start file reader
          const reader = new FileReader();
          
          reader.onload = e => {
            
            if (e.target.result) {
              load_img(e.target.result, "second_image")
              original_img_2 = e.target.result
                if (document.getElementById("second_image").childNodes[0].nodeName == "IMG"){
                  
                  $.ajax({
                    type: 'POST',
                    url: 'http://127.0.0.1:5000/hybrid_req',
                    data: {
                      img_upload1: original_img_1,
                      img_upload2: original_img_2,
                      D0Low:D0Low.value,
                      D0High:D0High.value
                    },
                    success: function (res) {
                        var responce = JSON.parse(res)
                        console.log(responce)
                        load_img(responce["1"],'result1')
                        load_img(responce["2"],'result2')
                      
                    }
              
              
                  })}
            }
          };
          reader.readAsDataURL(e.target.files[0]);
        }
      });
  
})
function load_img(src , id){
  let imgdiv = document.getElementById(id)
  
  let img = document.createElement('img');
  img.src = src;
  img.style.cssText=`
  height: 300px;
  padding: 0;
  margin: -24px;
  position: relative;
  top: -18px;
  left: 10px;
  `
  // clean result before
  imgdiv.innerHTML = '';
  // append new image
  imgdiv.appendChild(img)
}

//start upload button
image_input1.addEventListener('change', e => {
  e.preventDefault()
  if (e.target.files.length) {
      // start file reader
      const reader = new FileReader();
      reader.onload = e => {
        
        if (e.target.result) {
   
          load_img(e.target.result, "first_image")
          original_img_1 = e.target.result
          if (document.getElementById("second_image").childNodes[0].nodeName == "IMG"){

              
              $.ajax({
                type: 'POST',
                url: 'http://127.0.0.1:5000/hybrid_req',
                data: {
                  img_upload1: original_img_1,
                  img_upload2: original_img_2,
                  D0Low:D0Low.value,
                  D0High:D0High.value
                  
                },
                success: function (res) {
                    var responce = JSON.parse(res)
                    console.log(responce)
                    load_img(responce["1"],'result1')
                    load_img(responce["2"],'result2')
                  
                }
          
          
              })
        }
        }
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });

  image_input2.addEventListener('change', e => {
    e.preventDefault()
    if (e.target.files.length) {
        // start file reader
        const reader = new FileReader();
        
        reader.onload = e => {
          
          if (e.target.result) {
            load_img(e.target.result, "second_image")
            original_img_2 = e.target.result
              if (document.getElementById("second_image").childNodes[0].nodeName == "IMG"){
                
                $.ajax({
                  type: 'POST',
                  url: 'http://127.0.0.1:5000/hybrid_req',
                  data: {
                    img_upload1: original_img_1,
                    img_upload2: original_img_2,
                    D0Low:D0Low.value,
                    D0High:D0High.value
                  },
                  success: function (res) {
                      var responce = JSON.parse(res)
                      console.log(responce)
                      load_img(responce["1"],'result1')
                      load_img(responce["2"],'result2')
                    
                  }
            
            
                })}
          }
        };
        reader.readAsDataURL(e.target.files[0]);
      }
    });
//end upload button

//getting the uploaded image from the backend so it won't be lost if we go from one directory to another
//the photo is saved in a session state in the backend 

//let current_img_src = document.getElementById("current_img_source").innerHTML
