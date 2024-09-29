
var myVar;


var body = document.getElementsByTagName("body")[0];

body.addEventListener("load", timeout(), false);


function timeout() {
  myVar = setTimeout(showPage, 1000);
}

function showPage() {
    $('#loader').animate({opacity: 0}, 1000);
    $('#map').animate({opacity: 1}, 1000);
}
