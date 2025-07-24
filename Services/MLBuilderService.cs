using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using ChatBotMLNET.Models;
using static Microsoft.ML.DataOperationsCatalog;
using static SymSpell;

namespace ChatBotMLNET.Services;

public static class MLBuilderService
{
    private static string basePath = AppDomain.CurrentDomain.BaseDirectory;
    public static string PathDataSet { get; set; } =  Path.Combine(basePath, "Resource", "Domande_1.csv");
    public static string PathWord2Vec { get; set; } = Path.Combine(basePath, "Resource", "VocabolarioVettoriale", "cc.it.300.vec");
    public static string PathDictionary { get; set; } =  Path.Combine(basePath, "Resource", "DizionarioIT-100k.txt");

    public static readonly MLContext mlContext = new();
    public static IDataView data { get; set; }
    public static ITransformer Model { get; set; }
    public static PredictionEngine<QuestionAnswer, QuestionAnswerPrediction> predictionEngine { get; set; }
    static SymSpell symSpell = new SymSpell();
    public static TrainTestData trainTestSplit { get; set; }
    public static Word2VecService Word2Vec { get; set; } = new Word2VecService(PathWord2Vec);
    public static List<QuestionAnswer> AllDataSet { get; set; } = mlContext.Data.CreateEnumerable<QuestionAnswer>(LoadDataCSV(PathDataSet), reuseRowObject: false).ToList();

    public static IEstimator<ITransformer> BuildAdvancedPipeline() => 
        mlContext.Transforms.Text.NormalizeText(
                outputColumnName: "NormalizedText",
                inputColumnName: "Question")

            // Correzione ortografica con SymSpell
            .Append(mlContext.Transforms.CustomMapping<TextDataInput, TextDataCorrected>(
                ApplySymSpellCorrection, "SymSpellCorrection"))

            // Tokenizzazione e rimozione stopwords
            .Append(mlContext.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "CorrectedText"))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                outputColumnName: "CleanTokens",
                inputColumnName: "Tokens",
                language: StopWordsRemovingEstimator.Language.Italian))

            // Word Embeddings e N-gram
            .Append(mlContext.Transforms.Text.ProduceWordBags(
                outputColumnName: "WordBags",
                inputColumnName: "CleanTokens",
                ngramLength: 3,
                useAllLengths: true,
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf))
            .Append(mlContext.Transforms.Text.ProduceWordBags(
                outputColumnName: "Features",
                inputColumnName: "CleanTokens",
                ngramLength: 2,
                useAllLengths: true,
                weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf))

            // Character N-grams
            .Append(mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys(
                outputColumnName: "CharTokens",
                inputColumnName: "CorrectedText"))
            .Append(mlContext.Transforms.Text.ProduceNgrams(
                outputColumnName: "CharNgrams",
                inputColumnName: "CharTokens",
                ngramLength: 3,
                useAllLengths: false,
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf))

            // Semantic Embedding handling
            .Append(mlContext.Transforms.CopyColumns(
                outputColumnName: "SemanticEmbedding",
                inputColumnName: "Features"))

            // Concatenate all features
            .Append(mlContext.Transforms.Concatenate(
                outputColumnName: "AllFeatures",
                "Features", "WordBags", "CharNgrams", "SemanticEmbedding"))
            .AppendCacheCheckpoint(mlContext)

            // Mappatura label
            .Append(mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label",
                inputColumnName: "Answer"))

            // Trainer
            .Append(mlContext.MulticlassClassification.Trainers.LightGbm(
                labelColumnName: "Label",
                featureColumnName: "AllFeatures",
                numberOfLeaves: 64,
                numberOfIterations: 1000,
                minimumExampleCountPerLeaf: 10))

            // Mappatura risposta finale
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedAnswer",
                inputColumnName: "PredictedLabel"));

    // Nuovo metodo per creare una pipeline con fallback semantico integrato
    public static IEstimator<ITransformer> BuildEnhancedPipeline(Word2VecService word2VecService)
    {
        var pipeline = BuildAdvancedPipeline();

        // Aggiungiamo un transform personalizzato che calcola anche vettori Word2Vec
        pipeline = pipeline.Append(mlContext.Transforms.CustomMapping<QuestionAnswerPrediction, EnhancedPrediction>(
            (input, output) => {
                // Copiamo i campi standard
                output.PredictedAnswer = input.PredictedAnswer;
                output.SimilarityScore = input.SimilarityScore;

                // Aggiungiamo campi extra da Word2Vec
                if (output.InputQuestion != null)
                {
                    output.W2VScore = CalculateW2VScore(output.InputQuestion, word2VecService);
                }
            }, "SemanticScoreCalculator"));

        return pipeline;
    }

    private static float CalculateW2VScore(string input,Word2VecService word2VecService)
    {
        IEnumerable<string> allQuestions = AllDataSet.Select(qa => qa.Question).Distinct();
        // 1) Embedding dell’input
        var inputVec = UtilityNLP.GetSentenceEmbedding(word2VecService, input);
        if (inputVec == null || inputVec.All(v => v == 0f))
            return 0f;

        float maxScore = 0f;
        // 2) Per ogni domanda nel dataset, calcola embedding e cosine similarity
        foreach (var question in allQuestions)
        {
            var qVec = UtilityNLP.GetSentenceEmbedding(word2VecService, question);
            if (qVec == null || qVec.All(v => v == 0f))
                continue;

            float score = UtilityNLP.CosineSimilarity(inputVec, qVec);
            if (score > maxScore)
                maxScore = score;
        }

        return maxScore;
    }

    // Nuovo metodo di predizione integrato con semantica Word2Vec DA COMPLETATARE
    public static ChatModel PredictWithSemanticFallback(string question, float confidenceThreshold = 0.699f)
    {
        if (predictionEngine == null || Word2Vec == null)
            throw new InvalidOperationException("Prediction engine or Word2Vec service not initialized");

        var result = new ChatModel();

        // 1. Correzione ortografica
        string correctedQuestion = UtilityNLP.CorreggiFrase(question,
            Path.Combine(AppContext.BaseDirectory, "Resource", "DizionarioIT-100k.txt"));

        // 2. Predizione ML.NET
        var input = new QuestionAnswer { Question = correctedQuestion };
        var prediction = predictionEngine.Predict(input);

        // 3. Calcolo score semantico con Word2Vec
        var questionVec = UtilityNLP.GetSentenceEmbedding(Word2Vec, correctedQuestion);

        // 4. Score ML.NET e decisione
        var mlScoreMax = prediction?.SimilarityScore?.Max() ?? 0f;
        result.SimilarityScoreML = mlScoreMax;

        // 5. Se lo score ML.NET è sufficiente, usa quella risposta
        if (mlScoreMax >= confidenceThreshold)
        {
            result.Question = correctedQuestion;
            result.Answer = prediction.PredictedAnswer;

            // Trova il contesto originale
            var matchingQA = AllDataSet?.FirstOrDefault(qa => qa.Answer == prediction.PredictedAnswer);
            if (matchingQA != null)
                result.Context = matchingQA.Context;

            return result;
        }

        // 6. Altrimenti usa Word2Vec per fallback semantico
        var semanticResult = UtilityNLP.FindMostSimilarQuestion(correctedQuestion, AllDataSet, Word2Vec);
        if (semanticResult != null)
        {
            return semanticResult;
        }

        // 7. Se proprio non troviamo nulla
        result.Answer = "Non ho capito la domanda. Puoi riformularla?";
        return result;
    }

    public static ITransformer Train()
    {
        var pipeline = BuildAdvancedPipeline();
        var model = pipeline.Fit(trainTestSplit.TrainSet);
        predictionEngine = mlContext.Model.CreatePredictionEngine<QuestionAnswer, QuestionAnswerPrediction>(model);
        return Model = model;
    }

    public static IDataView LoadDataCSV(string dataSet)
    {
        data = mlContext.Data.LoadFromTextFile<QuestionAnswer>(dataSet, ',', true, allowQuoting: true);
        symSpell.LoadDictionary(Path.Combine(AppContext.BaseDirectory, "Resource", "DizionarioIT-100k.txt"), termIndex: 0, countIndex: 1);
        setTrainTestSplit();

        // Salva il dataset in memoria per l'uso in semantic fallback
        return data;
    }

    public static TrainTestData setTrainTestSplit() => trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

    public static double CrossValidate()
    {
        var pipeline = BuildAdvancedPipeline();
        var cvResults = mlContext.MulticlassClassification
            .CrossValidate(data: trainTestSplit.TrainSet,
                           estimator: pipeline,
                           numberOfFolds: 5,
                           labelColumnName: "Label");
        // Media delle MacroAccuracy sui 5 fold
        var avg = cvResults.Average(r => r.Metrics.MacroAccuracy);
        Console.WriteLine($"5‑fold CV MacroAccuracy media: {avg:P2}");
        return avg;
    }

    public static double EvaluateModel(ITransformer model)
    {
        var predictions = model.Transform(trainTestSplit.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
        return metrics.MacroAccuracy;
    }

    // Funzione static per la correzione
    public static void ApplySymSpellCorrection(TextDataInput input, TextDataCorrected output)
    {
        output.CorrectedText = symSpell.Lookup(input.NormalizedText, Verbosity.Closest, maxEditDistance: 2)
                                    .FirstOrDefault()?.term ?? input.NormalizedText;
    }
}

// Classe per gestire predizioni potenziate
public class EnhancedPrediction : QuestionAnswerPrediction
{
    public string InputQuestion { get; set; }
    public float W2VScore { get; set; }
    public bool UsedSemanticFallback { get; set; }
}