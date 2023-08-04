

// var t = 1;


function autoplay() {
    if (document.getElementById("autoplayCheckbox").checked == true){
        window.autoplayInterval = setInterval(stepTime, 333);
    } else {
        clearInterval(window.autoplayInterval);
    }
}

function stepTime() {
    t = getStep();
    if (t == document.getElementById("t_slider").max) {
        setStep(1);
    } else {
        setStep(t + 1);
    }
}

function getStep() {
    return parseInt(document.getElementById("t_slider").value);
}
function setStep(t) {
    document.getElementById("t_slider").value = t;
    document.getElementById("show_t").innerHTML = t;
    // var path = "imgs/img_" + t + ".png";
    for (elem of document.getElementsByClassName("anim")) {
        elem.src = elem.src.split("___T")[0] + "___T" + t + ".png";
        console.log(elem.src);
    }
}

setStep(1);
document.getElementById("t_slider").oninput = function() {
  setStep(this.value);
}

autoplay();




// make a list of imgs/img_1.png imgs/img_2.png imgs/img_3.png etc
// var imgList = [];
// for (var i = 1; i <= 19; i++) {
//     imgList.push("imgs/img_" + i + ".png");
// }

