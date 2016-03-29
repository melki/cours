$(document).ready(function() {
		 /*$.ajax({
		    type : "POST",
		    url : "resolution",
		    data: JSON.stringify(this.id, null, '\t'),
		    contentType: 'application/json;charset=UTF-8',
		    success: function(result) {
		        printResult(result);
		    }*/

		 $('.ia').click(function(){
		 	var taille = +prompt("Taille du plateau",4);
		 	console.log(taille);
		 	if(this.id==c1)
		 	{
		 		playAlleatoire();
		 	}
		 })

		 function playAlleatoire()
		 {
		 	
		 }
		
});