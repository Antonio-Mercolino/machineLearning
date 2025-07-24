using ChatBotMLNET.Models;
using static ChatBotMLNET.Services.UtilityNLP;
using static ChatBotMLNET.Services.MLBuilderService;
using Microsoft.ML;
using ChatBotMLNET.Services;

namespace ChatBotMLNET;

public class ManagementNLP
{
    public readonly ITransformer transformer = Train();

    public ChatModel ResponseBot(string domanda)
    {
        if (transformer == null)
            return new ChatModel { Answer = "Modello non inizializzato" };
        // Utilizziamo il nuovo metodo integrato che gestisce sia ML.NET che fallback Word2Vec
        return MLBuilderService.PredictWithSemanticFallback(domanda);
    }

    // Metodo alternativo che conserva la vecchia logica per compatibilità
    public ChatModel ResponseBotLegacy(string domanda)
    {
        if (transformer == null)
            return new ChatModel { Answer = "Modello non inizializzato" };

        var model = new ChatModel();
        string parolaCorretta = CorreggiFrase(domanda, PathDictionary,Word2VecService._wordVectors);
        //if (CosineSimilarity(GetSentenceEmbedding(Word2Vec, parolaCorretta), GetSentenceEmbedding(Word2Vec, domanda)) > 0.8)
        //    domanda = parolaCorretta;

        double similarity = CosineSimilarity(GetSentenceEmbedding(Word2Vec, parolaCorretta), GetSentenceEmbedding(Word2Vec, domanda));

        if (similarity > 0.8 && parolaCorretta.Length <= domanda.Length + 2)
        {
            // Esegui sostituzione solo se parolaCorretta è plausibilmente simile in lunghezza
            domanda = parolaCorretta;
        }

        var input = new QuestionAnswer { Question = domanda };
        var prediction = predictionEngine.Predict(input);
        var predictedAnswer = prediction.PredictedAnswer;

        var matchingQA = AllDataSet.FirstOrDefault(qa => qa.Answer == predictedAnswer);
        if (matchingQA != null)
        {
            var inputVec = GetSentenceEmbedding(Word2Vec, domanda);
            var originalQuestionVec = GetSentenceEmbedding(Word2Vec, matchingQA.Question);

            if (inputVec != null && originalQuestionVec != null)
            {
                var questionSimilarity = prediction?.SimilarityScore?.Max();
                model.SimilarityScoreML = questionSimilarity ?? 0f;

                if (questionSimilarity < 0.47f)
                {
                    var bestQA = FindMostSimilarQuestion(domanda, AllDataSet, Word2Vec);
                    if (bestQA != null)
                    {
                        if(bestQA.Answer == "" && bestQA.Ambigue.Count == 0)
                            model.Answer = "Mi spiace, ma non posso ancora ripondere a questa domanda";
                        else
                            model = bestQA;
                    }
                    else
                        model.Answer = "Non ho capito, puoi riformulare la frase";
                }
                else
                {
                    model.Answer = predictedAnswer;
                }
                return model;
            }
        }

        return model;
    }
}