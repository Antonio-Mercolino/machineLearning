# ChatBotMLNET
## ChatBot con ML.NET per la creazione di un NLP

per integrarlo in un progetto esistente, basta includere il progetto ChatBotMLNET come riferimento instanziare un oggetto di tipo ManagementNLP ES: private ManagementNLP? managementNLP = new ManagementNLP(); e richiamre la funzione ResponseBotLegacy(message) 
passando come argomento il messaggio.

Esempio di utilizzo in un  progetto console:
```csharp
	using System;
	using ChatBotMLNET;

	class Program
	{
		static void Main(string[] args)
		{
		    	ManagementNLP? managementNLP = new ManagementNLP();
			Console.WriteLine("Scrivi un messaggio per il chatbot:");
			string message = Console.ReadLine();
			string response = managementNLP.ResponseBotLegacy(message);
			Console.WriteLine("Risposta del chatbot: " + response);
		}
	}
```
Il chatbot risponderà in base al messaggio fornito. Puoi continuare a interagire con il chatbot inserendo nuovi messaggi.
Puoi anche personalizzare il chatbot aggiungendo nuove frasi e risposte, assicurandoti di mantenere il formato corretto nel file Domande.csv all'interno del path: /Resource/, il modello verrà addestrato automaticamente all'avvio dell'applicazione. 

NB: il file vocabolario vettoriale cc.it.300.vec è troppo grande per poterlo committare quindi va scaricato a parte e inserito sotto la cartella /Resource/VocabolarioVettoriale/
