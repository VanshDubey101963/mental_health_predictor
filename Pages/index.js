document.getElementById('userInputForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get user input values
    let userInput = {
        'Gender': document.getElementById('gender').value.trim(),
        'Country': document.getElementById('country').value.trim(),
        'Occupation': document.getElementById('occupation').value.trim(),
        'self_employed': document.getElementById('selfEmployed').value.trim(),
        'family_history': document.getElementById('familyHistory').value.trim(),
        'treatment': document.getElementById('treatment').value.trim(),
        'Days_Indoors': document.getElementById('daysIndoors').value.trim(),
        'Growing_Stress': document.getElementById('growingStress').value.trim(),
        'Changes_Habits': document.getElementById('changesHabits').value.trim(),
        'Mental_Health_History': document.getElementById('mentalHealthHistory').value.trim(),
        'Mood_Swings': document.getElementById('moodSwings').value.trim(),
        'Coping_Struggles': document.getElementById('copingStruggles').value.trim(),
        'mental_health_interview': document.getElementById('mentalHealthInterview').value.trim(),
        'Social_Weakness': document.getElementById('socialWeakness').value.trim(),
        'Work_Interest': document.getElementById('workInterest').value.trim()
    };

    fetch('http://127.0.0.1:5000/',{
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({input:userInput})
    }).then(
        response => response.json()
    ).then(
        data => document.getElementById('predictionResult').innerText = `Predicted Care Option: ${data['prediction']}`
    )
    .catch(error =>{
        console.error("error:",error)
    })
});