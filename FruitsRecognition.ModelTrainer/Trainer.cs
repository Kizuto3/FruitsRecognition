using FruitsRecognition.MLContextFactory;
using FruitsRecognition.MLContextFactory.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FruitsRecognition.ModelTrainer
{
    public class Trainer
    {
        private static readonly string _predictedLabelColumnName = "PredictedLabel";
        private static readonly string _keyColumnName = "LabelAsKey";

        private ImageRecognitionContext _imageRecognitionModel;

        public MLContext Context { get; private set; }

        public IDataView TrainData { get; private set; }

        public Trainer()
        {
            _imageRecognitionModel = new ImageRecognitionContext();
        }

        public TransformerChain<KeyToValueMappingTransformer> Train(string pathToData, int seed)
        {
            Context = _imageRecognitionModel.Create(seed);
            var input = LoadLabeledImagesFromPath(pathToData);
            var data = _imageRecognitionModel.GetDataView(Context, input);

            // Load the images and convert the labels to keys to serve as categorical values
            var images = Context.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Input.Label), outputColumnName: _keyColumnName)
                .Append(Context.Transforms.LoadRawImageBytes(inputColumnName: nameof(Input.ImagePath), outputColumnName: nameof(Input.Image), imageFolder: pathToData))
                .Fit(data).Transform(data);

            // Split the dataset for training and testing
            var trainTestData = _imageRecognitionModel.SplitData(Context, images, 0.2, 1);
            TrainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Create an image-classification pipeline and train the model
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = nameof(Input.Image),
                LabelColumnName = _keyColumnName,
                ValidationSet = testData,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // Pretrained DNN
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false
            };

            var pipeline = _imageRecognitionModel.GeneratePipeline(Context, options, _predictedLabelColumnName);

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(TrainData);

            // Evaluate the model and show the results
            var predictions = model.Transform(testData);

            return model;
        }

        public void ExportModel(TransformerChain<KeyToValueMappingTransformer> model, string savePath)
        {
            Context.Model.Save(model, TrainData.Schema, savePath);
        }

        private List<Input> LoadLabeledImagesFromPath(string path)
        {
            var images = new List<Input>();
            var directories = Directory.EnumerateDirectories(path);

            var files = Directory.EnumerateFiles(path);

            images.AddRange(files.Select(x => new Input
            {
                ImagePath = Path.GetFullPath(x),
                Label = Path.GetFileName(path)
            }));

            foreach (var directory in directories)
            {
                images.AddRange(LoadLabeledImagesFromPath(directory));
            }

            return images;
        }
    }
}
