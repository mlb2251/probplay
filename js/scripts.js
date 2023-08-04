

var t = 1;
var tMax;
var autoplayOn = true;
var autoplayInterval = undefined;


function refreshTMax() {
    for (elem of document.getElementsByClassName("slider_t")) {
        elem.min = 1;
        elem.max = tMax;
    }
}

function refresh() {
    for (elem of document.getElementsByClassName("show_t")) {
        elem.innerHTML = t;
    }
    for (elem of document.getElementsByClassName("slider_t")) {
        elem.value = t;
        elem.min = 1;
        elem.max = tMax;
    }
    for (elem of document.getElementsByClassName("anim")) {
        elem.src = elem.src.split("___T")[0] + "___T" + t + ".png";
    }
    for (elem of document.getElementsByClassName("autoplayCheckbox")) {
        elem.checked = autoplayOn;
    }
    if (autoplayOn === true && autoplayInterval === undefined) {
        autoplayInterval = setInterval(stepTime, 333);
    }
    if (autoplayOn === false && autoplayInterval !== undefined) {
        autoplayInterval = clearInterval(autoplayInterval);
    }
}

function stepTime() {
    t = (t % tMax) + 1;
    refresh();
}
function stepTimeBack() {
    t -= 1;
    if (t < 1) {
        t = tMax;
    }
    refresh();
}


document.onkeydown = function (e) {
    var key = e.key.toLowerCase()

    if (key == 'a') {
        stepTimeBack();
    }
    else if (key == 'd') {
        stepTime();
    }
    else if (key == 'e') {
        toggleAutoplay();
    }
    // console.log(e.key);
}

function toggleAutoplay() {
    autoplayOn = !autoplayOn;
    refresh();
}



// set stuff up
window.onload = function() {

    for (elem of document.getElementsByClassName("slider_t")) {
        elem.oninput = function() {
            t = parseInt(this.value);
            refresh();
        }
    }

    for (elem of document.getElementsByClassName("autoplayCheckbox")) {
        elem.oninput = function() {
            autoplayOn = this.checked;
            refresh();
        }
    }

    refreshTMax();
    refresh();
}


