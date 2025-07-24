using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace ChatBotMLNET
{
    public class Word2VecService
    {
        public static Dictionary<string, float[]> _wordVectors = new();

        /// <summary>
        /// Costruttore: se esiste il file binario, lo carica, altrimenti lo crea da textModelPath e poi lo carica.
        /// </summary>
        /// <param name="textModelPath">Percorso del modello text (.txt)</param>
        /// <param name="binaryModelPath">Percorso desiderato per il .bin (opzionale)</param>
        public Word2VecService(string textModelPath, string? binaryModelPath = null)
        {
            if (string.IsNullOrEmpty(textModelPath) || !File.Exists(textModelPath))
                throw new ArgumentException("Percorso modello testo non valido.", nameof(textModelPath));

            // Se non specificato, affianca .bin al .txt
            var binPath = binaryModelPath
                          ?? Path.ChangeExtension(textModelPath, ".bin");

            if (!File.Exists(binPath))
            {
                // 1) Serializza una volta il modello testuale in binario
                SaveWord2VecBinary(textModelPath, binPath);
            }

            // 2) Carica sempre dal binario
            _wordVectors = LoadWord2VecMemoryMappedSafe(binPath);
        }

        /// <summary>
        /// Restituisce l'embedding medio della frase.
        /// </summary>
        public float[] GetSentenceEmbedding(string sentence, int embeddingDim = 300)
        {
            if (string.IsNullOrWhiteSpace(sentence))
                return new float[embeddingDim];

            // Tokenizzazione semplice
            char[] delimiters = { ' ', ',', '.', '?', '!', ';', ':' };
            var tokens = sentence
                         .ToLowerInvariant()
                         .Split(delimiters, StringSplitOptions.RemoveEmptyEntries);

            var validVectors = new List<float[]>();
            foreach (var token in tokens)
            {
                var vec = GetWordVector(token);
                if (vec != null)
                    validVectors.Add(vec);
            }

            if (validVectors.Count == 0)
                return new float[embeddingDim];

            var avg = new float[embeddingDim];
            for (int i = 0; i < embeddingDim; i++)
                avg[i] = validVectors.Select(v => v[i]).Average();

            return avg;
        }

        /// <summary>
        /// Restituisce l'embedding medio della frase (versione alternativa).
        /// </summary>
        public float[]? GetAverageVector(string sentence)
        {
            if (string.IsNullOrWhiteSpace(sentence))
                return null;

            var words = sentence
                        .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            var vectors = new List<float[]>();
            foreach (var w in words)
            {
                if (_wordVectors.TryGetValue(w.ToLowerInvariant(), out var vec))
                    vectors.Add(vec);
            }

            if (vectors.Count == 0)
                return null;

            int dim = vectors[0].Length;
            var sum = new float[dim];
            foreach (var vec in vectors)
                for (int i = 0; i < dim; i++)
                    sum[i] += vec[i];

            for (int i = 0; i < dim; i++)
                sum[i] /= vectors.Count;

            return sum;
        }

        /// <summary>
        /// Restituisce il vettore di una singola parola, se presente.
        /// </summary>
        public float[]? GetWordVector(string word)
        {
            if (word == null) return null;
            _wordVectors.TryGetValue(word, out var vec);
            return vec;
        }

        #region -- Parsing veloce da testo (mono-thread) --

        private static Dictionary<string, float[]> LoadWord2VecFast(string modelPath)
        {
            var dict = new Dictionary<string, float[]>(capacity: 2_000_000);

            using var fs = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20);
            using var reader = new StreamReader(fs, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 20);

            string? line;
            bool skipFirst = true;

            while ((line = reader.ReadLine()) != null)
            {
                if (skipFirst)
                {
                    skipFirst = false;
                    continue;
                }

                int sep = line.IndexOf(' ');
                if (sep < 0) continue;

                string word = line[..sep];
                var span = line.AsSpan(sep + 1);
                var vector = new float[300];

                int idx = 0, start = 0;
                for (int i = 0; i <= span.Length && idx < 300; i++)
                {
                    if (i == span.Length || span[i] == ' ')
                    {
                        var token = span.Slice(start, i - start);
                        if (float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out float v))
                            vector[idx++] = v;
                        else
                            break;
                        start = i + 1;
                    }
                }

                if (idx == 300)
                    dict[word] = vector;
            }

            return dict;
        }

        #endregion

        #region -- Binary Serialization & Deserialization --

        /// <summary>
        /// Serializza il modello testuale in un file binario.
        /// </summary>
        public static void SaveWord2VecBinary(string textModelPath, string binaryPath)
        {
            var wordVectors = LoadWord2VecFast(textModelPath);

            using var fs = new FileStream(binaryPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 20);
            using var writer = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

            writer.Write(wordVectors.Count);
            foreach (var kv in wordVectors)
            {
                byte[] wb = Encoding.UTF8.GetBytes(kv.Key);
                writer.Write(wb.Length);
                writer.Write(wb);

                var vec = kv.Value;
                for (int i = 0; i < vec.Length; i++)
                    writer.Write(vec[i]);
            }
        }

        /// <summary>
        /// Carica il modello dal file binario, in ~15–20 s per 4 GB.
        /// </summary>

        public static Dictionary<string, float[]> LoadWord2VecMemoryMappedSafe(string binaryPath)
        {
            var dict = new Dictionary<string, float[]>(capacity: 2_000_000);

            using var mmf = MemoryMappedFile.CreateFromFile(binaryPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

            // 1) Leggo il count
            int count = reader.ReadInt32();

            for (int i = 0; i < count; i++)
            {
                // 2) parola
                int wordLen = reader.ReadInt32();
                var wb = reader.ReadBytes(wordLen);
                string word = Encoding.UTF8.GetString(wb);

                // 3) vettore in blocco
                int byteCount = 300 * sizeof(float);
                var data = reader.ReadBytes(byteCount);
                var vector = new float[300];
                Buffer.BlockCopy(data, 0, vector, 0, byteCount);

                dict[word] = vector;
            }

            return dict;
        }
    }


    #endregion


}
