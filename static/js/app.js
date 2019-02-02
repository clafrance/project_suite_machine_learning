// function buildMetadata(sample) {
//   var url = "/metadata/" + sample;
//   d3.json(url).then(function (response) {
//     console.log(response);
//     d3.select("#metadata")
//       .selectAll("div").remove();
//     d3.select("#metadata")
//       .selectAll("div")
//       .data(Object.entries(response))
//       .enter()
//       .append("div")
//       .text(function (d) {
//         return d[0] + ': ' + d[1];
//       });
//   });
// }



// var submit = d3.select("#submit1");

// submit.on("click", function () {
//     // Prevent the page from refreshing
//     d3.event.preventDefault();

//     // Select the input element and get the raw HTML node
//     var inputValue1 = d3.select("#input1").property("value");
//     var inputValue2 = d3.select("#input2").property("value");
//     var inputValue3 = d3.select("#input3").property("value");

//     var names = [inputValue1, inputValue2, inputValue3];
//     plot(names);
//     console.log(names);
// });



var trace1 = {
    labels: ["Median gross rent", "Per Capita Income", "Population Density", "House Age", "Household Income", "Population",
       "Median Age", "Poverty Rate", "Unemployment Rate", "Advanced Degree Hoders", "pop_arc/eng"],
    values: [0.51, 0.2520, 0.0399, 0.0329, 0.2568, 0.0135, 0.031, 0.026, 0.0314, 0.0195, 0.017],
    type: "pie"
 };
var layout = {
    title: "Model Factors Weight Chart"
};
     
var data = [trace1];
Plotly.newPlot("pieChart", data, layout);
console.log("Plot Pie Chart");




// // Initialize the dashboard
// init();
