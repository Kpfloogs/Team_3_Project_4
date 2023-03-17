const url = "https://team-3-project-4.onrender.com/api/v1.0/brain_stroke_data"

// looking at arrays in the console log:
// d3.json(url).then(function(data) {
//     console.log(data);
//   });

d3.json(url).then(function(data){
    let trace1 = {
        x: data.map(row => row.gender),
        y: data.map(row => row.bmi),
        type:'bar'}
      
    let traceData1 = [trace1];

    let layout1 = {
            title: "BMI by Gender"
        };
        
        Plotly.newPlot("bar", traceData1, layout1)
    
    let trace2 = {
        x: data.map(row => row.age),
        y: data.map(row => row.smoking_status),
        type:'bar'}
    
    let traceData2 = [trace2];

    let layout2 = {
        title: "BMI by Gender"
        }; 

        Plotly.newPlot("bubble", traceData2, layout2)
    
    let trace3 = {
        x: data.map(row => row.gender),
        y: data.map(row => row.bmi),
        type:'bar'}
                    
    let traceData3 = [trace3];
            
    let layout = {
        title: "BMI by Gender"
        };
                    
        Plotly.newPlot("gauge", traceData3, layout)

});





