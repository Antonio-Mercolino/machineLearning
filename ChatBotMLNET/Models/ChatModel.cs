namespace ChatBotMLNET.Models;

public class ChatModel
{
    public float SimilarityScoreML { get; set; }
    public float SimilarityScoreW2V { get; set; }
    public Dictionary<string, string> Ambigue { get; set; } = new();
    public string Question { get; set; } = string.Empty;
    public string Answer { get; set; } = string.Empty;
    public string Context { get; set; } = string.Empty;
}
