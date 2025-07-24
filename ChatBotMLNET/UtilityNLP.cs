using ChatBotMLNET.Models;
using static SymSpell;

namespace ChatBotMLNET.Services;

public static class UtilityNLP
{
    /// <summary>
    /// Calcola la distanza di Levenshtein tra due stringhe.
    /// </summary>
    /// <param name="s">Prima stringa.</param>
    /// <param name="t">Seconda stringa.</param>
    /// <returns>Numero minimo di operazioni (inserimento, cancellazione, sostituzione) per trasformare s in t.</returns>
    public static int LevenshteinDistance(string s, string t)
    {
        if (string.IsNullOrEmpty(s))
            return string.IsNullOrEmpty(t) ? 0 : t.Length;
        if (string.IsNullOrEmpty(t))
            return s.Length;

        int n = s.Length;
        int m = t.Length;
        var d = new int[n + 1, m + 1];

        // Inizializza primi valori
        for (int i = 0; i <= n; i++)
            d[i, 0] = i;
        for (int j = 0; j <= m; j++)
            d[0, j] = j;

        // Calcolo matriciale
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;

                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1,        // cancellazione
                             d[i, j - 1] + 1),       // inserimento
                             d[i - 1, j - 1] + cost); // sostituzione
            }
        }

        return d[n, m];
    }
    public static float[] GetWordEmbedding(Dictionary<string, float[]> word2Vec, string parola)
    {
        parola = parola.ToLowerInvariant();
        if (word2Vec.TryGetValue(parola, out var vettore))
            return vettore;

        // Se parola non trovata, restituisci un vettore nullo
        return new float[word2Vec.First().Value.Length];
    }

    public static float[] GetSentenceEmbedding(Word2VecService word2VecModel, string sentence, int embeddingDim = 300)
    {
        // Converte la frase in minuscolo e tokenizza rimuovendo punteggiatura di base
        char[] delimiters = new char[] { ' ', ',', '.', '?', '!', ';', ':' };
        var tokens = sentence.ToLower().Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

        List<float[]> validVectors = new List<float[]>();
        foreach (var token in tokens)
        {
            var vec = word2VecModel.GetWordVector(token);
            if (vec != null)
                validVectors.Add(vec);
        }

        if (validVectors.Count == 0)
        {
            // Se nessun embedding viene trovato, ritorna un vettore di zeri (o potresti gestire diversamente)
            return new float[embeddingDim];
        }
        else
        {
            float[] avgVector = new float[embeddingDim];
            for (int i = 0; i < embeddingDim; i++)
            {
                avgVector[i] = validVectors.Select(v => v[i]).Average();
            }
            return avgVector;
        }
    }
    public static float[] GetSentenceEmbedding(Dictionary<string, float[]> word2Vec, string frase)
    {
        var parole = frase.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int dimensione = word2Vec.First().Value.Length;
        float[] somma = new float[dimensione];
        int conteggio = 0;

        foreach (var parola in parole)
        {
            var vettore = GetWordEmbedding(word2Vec, parola);
            if (vettore.Any(x => x != 0)) // escludi parole fuori vocabolario
            {
                for (int i = 0; i < dimensione; i++)
                    somma[i] += vettore[i];
                conteggio++;
            }
        }

        if (conteggio == 0)
            return new float[dimensione]; // fallback

        for (int i = 0; i < dimensione; i++)
            somma[i] /= conteggio;

        return somma;
    }
    public static bool HasNonZero(float[] array) => array.Any(value => value != 0.0f);
    

    public static ChatModel? FindMostSimilarQuestion(string inputQuestion, List<QuestionAnswer> dataset, Word2VecService model)
    {
        var inputVec = GetSentenceEmbedding(model, inputQuestion);
        if (inputVec == null || !HasNonZero(inputVec)) return null;
        var response = new ChatModel();

        // Step 1: trova candidati con soglia

        var candidati = dataset
          .Select(qa => new { QA = qa, Vec = GetSentenceEmbedding(model, qa.Question) })
          .Where(x => x.Vec != null)
          .Select(x => new { x.QA, Score = CosineSimilarity(inputVec, x.Vec) })
          .Where(x => x.Score > 0.40f)
          .OrderByDescending(x => x.Score)
          .ToList();

        // Raggruppa per contesto, e prendi il migliore per ognuno
        // Raccoglie i contesti distinti tra i top match (es. parlami di...)
        if (candidati.Any())
        {

            var best = candidati.First();
            var secondBestScore = candidati.Skip(1).FirstOrDefault()?.Score ?? 0f;


            var ambigue = candidati.Where(x => x.Score > 0.80f).GroupBy(x => x.QA.Context).Select(g => g.OrderByDescending(e => e.Score).First()).ToList();
            var score = (best.Score - secondBestScore);
            if (best.Score > 0.902f && score > 0.038f || best.Score >=0.99f)
            {
                response.Question = best.QA.Question;
                response.Answer = best.QA.Answer;
                response.Context = best.QA.Context;
                response.SimilarityScoreW2V = best.Score;
                return response;
            }
            else if (ambigue.Count > 1 && candidati.First().Score > 0.80f)
            {
                for (int i = 0; i < ambigue.Select(x => x.Score == best.Score && x.Score > 0.90).ToList().Count; i++)
                {
                    response.Ambigue.Add($"{i + 1}. {ambigue[i].QA.Context}", ambigue[i].QA.Answer);
                }
                return response;
            }
            else if (candidati.Where(x => x.Score > 0.40).FirstOrDefault()  != null)
            {
                response.Question = best.QA.Question;
                response.Answer = best.QA.Answer;
                response.Context = best.QA.Context;
                response.SimilarityScoreW2V = best.Score;
                return response;
            }
        }
        return response;

    }
    // Funzione per calcolare la similarità coseno tra due vettori
    public static float CosineSimilarity(float[] vec1, float[] vec2)
    {
        if (vec1.Length != vec2.Length)
            throw new ArgumentException("I vettori devono avere la stessa dimensione.");

        float dotProduct = vec1.Zip(vec2, (a, b) => a * b).Sum();
        double magnitude1 = Math.Sqrt(vec1.Sum(v => v * v));
        double magnitude2 = Math.Sqrt(vec2.Sum(v => v * v));
        return (float)(dotProduct / (magnitude1 * magnitude2));
    }
   public static string CorreggiFrase(string frase, string dicPathIT)
    {
        var symSpell = new SymSpell(82765, 2); // Max edit distance: 2

        // Carica dizionario da file
        symSpell.LoadDictionary(dicPathIT, termIndex: 0, countIndex: 1);

        var parole = frase.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        List<string> corrette = new();

        foreach (var parola in parole)
        {
            var suggerimenti = symSpell.Lookup(parola.ToLower(), Verbosity.Closest, 2);
            if (suggerimenti.Any())
                corrette.Add(suggerimenti.First().term);
            else
                corrette.Add(parola);
        }

        return string.Join(" ", corrette);
    }
    public static string CorreggiFrase(string frase, string dicPathIT, Dictionary<string, float[]> word2Vec)
    {
        var symSpell = new SymSpell(82765, 2);
        symSpell.LoadDictionary(dicPathIT, termIndex: 0, countIndex: 1);

        var parole = frase.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        List<string> corrette = new();

        foreach (var parola in parole)
        {
            var suggerimenti = symSpell.Lookup(parola.ToLower(), Verbosity.Closest, 2);
            if (suggerimenti.Any())
            {
                var correzione = suggerimenti.First().term;

                // Valuta similarità semantica parola-parola
                double sim = CosineSimilarity(
                    GetWordEmbedding(word2Vec, parola),
                    GetWordEmbedding(word2Vec, correzione));

                // Applica correzione solo se semanticamente simile
                if (sim > 0.7)
                    corrette.Add(correzione);
                else
                    corrette.Add(parola);
            }
            else
            {
                corrette.Add(parola);
            }
        }

        string fraseCorretta = string.Join(" ", corrette);

        // Confronta la frase originale con quella corretta
        double simFrasi = CosineSimilarity(
            GetSentenceEmbedding(word2Vec, frase),
            GetSentenceEmbedding(word2Vec, fraseCorretta));

        if (simFrasi > 0.85)
            return fraseCorretta;

        return frase;
    }

}
