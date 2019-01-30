// Array of names
const names = ["Jane", "John", "Jimbo", "Jedediah"];

// Slices first two names
const left = names.slice(0, 2);
console.log(left);

// Slices last two names and return
const right = names.slice(2, 4);
console.log(right);

//////////////////////////////////////////////////////////////////////////
// Sort the data array using the greekSearchResults value
data = [1, 2, 3, 4, 5, 6, 7, 8]
data.sort(function(a, b) {
  return parseFloat(b.greekSearchResults) - parseFloat(a.greekSearchResults);
});

// Slice the first 10 objects for plotting
data = data.slice(0, 10);

// Reverse the array due to Plotly's defaults
data = data.reverse();

// Trace1 for the Greek Data
var trace1 = {
  x: data.map(row => row.greekSearchResults),
  y: data.map(row => row.greekName),
  text: data.map(row => row.greekName),
  name: "Greek",
  type: "bar",
  orientation: "h"
};

// data
var data = [trace1];

// Apply the group bar mode to the layout
var layout = {
  title: "Greek gods search results",
  margin: {
    l: 100,
    r: 100,
    t: 100,
    b: 100
  }
};

// Render the plot to the div tag with id "plot"
Plotly.newPlot("plot", data, layout);
//////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////
// initialize the map on the "map" div with a given center and zoom
var map = L.map('map', {
	center: [51.505, -0.09],
	zoom: 13
});

//Marker
L.marker([50.5, 30.5]).addTo(map);


//Pop-up
var popup = L.popup()
	.setLatLng(latlng)
	.setContent('<p>Hello world!<br />This is a nice popup.</p>')
  .openOn(map);
  
//title layer
L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png?{foo}', {foo: 'bar'}).addTo(map);

