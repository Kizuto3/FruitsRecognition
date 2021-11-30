using FruitsRecognition.MLContextFactory.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace FruitsRecognition.Predictor
{
    public class PredictorModel
    {
        private string[] _labels;

        public PredictionEngine<Input, Output> Predictor { get; private set; }

        public void LoadModel(string modelPath, int seed)
        {
            // Load a trained image-classification model, create a
            // prediction engine, and get an ordered list of labels
            // for showing prediction results
            var context = new MLContext(seed: seed);
            var model = context.Model.Load(modelPath, out DataViewSchema schema);

            Predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            _labels = GetOrderedLabels(Predictor.OutputSchema);
        }

        public List<KeyValuePair<string, float>> Predict(Input input)
        {
            var prediction = Predictor.Predict(input);
            var label = prediction.PredictedLabel;

            // Show the results
            var result = new Dictionary<string, float>();

            for (int i = 0; i < _labels.Length; i++)
                result.Add(_labels[i], prediction.Score[i]);

            var list = result.ToList();
            list.Sort((pair1, pair2) => pair1.Value.CompareTo(pair2.Value) * -1);

            return list;
        }

        private static string[] GetOrderedLabels(DataViewSchema schema)
        {
            var buffer = new VBuffer<ReadOnlyMemory<char>>();
            schema.GetColumnOrNull("Score").Value.GetSlotNames(ref buffer);
            return buffer.DenseValues().Select(x => x.ToString()).ToArray();
        }
    }
}
