using Microsoft.ML.Data;

namespace ChatBotMLNET.Models;

public class QuestionAnswer
{
    [LoadColumn(0)]
    public string Question { get; set; } = string.Empty;
    [LoadColumn(2)]
    public string Answer { get; set; } = string.Empty;
    [LoadColumn(1)]
    public string Context { get; set; } = string.Empty;

/*    [VectorType(300)] // Lunghezza del vettore Word2Vec
    public float[] Features { get; set; } // Feature da usare nel modello*/

}
public class TextDataInput
{
    public string Question { get; set; }
    public string NormalizedText { get; set; }
}

public class TextDataCorrected : TextDataInput
{
    public string CorrectedText { get; set; }
}
