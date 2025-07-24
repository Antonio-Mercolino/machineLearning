using Microsoft.ML.Data;

namespace ChatBotMLNET.Models;
public class QuestionAnswerPrediction
{
    public string[]? PredictedQuestion { get; set; }
    public string PredictedAnswer { get; set; } = string.Empty;
    [ColumnName(@"Score")]
    public float[]? SimilarityScore { get; set; }   // Similarità tra domanda e risposta
}