using FruitsRecognition.MLContextFactory.Data;
using Microsoft.ML;
using Microsoft.ML.Vision;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;

namespace FruitsRecognition.MLContextFactory
{
    public class ImageRecognitionContext
    {
        public MLContext Create(int seed)
        {
            return new MLContext(seed: seed);
        }

        public IDataView GetDataView(MLContext context, List<Input> input)
        {
            // Create a DataView containing the image paths and labels
            var data = context.Data.LoadFromEnumerable(input);
            data = context.Data.ShuffleRows(data);

            return data;
        }

        public (IDataView TrainSet, IDataView TestSet) SplitData(MLContext context, IDataView images, double testFraction, int seed)
        {
            var trainTestData = context.Data.TrainTestSplit(images, testFraction: testFraction, seed: seed);
            return (trainTestData.TrainSet, trainTestData.TestSet);
        }

        public EstimatorChain<KeyToValueMappingTransformer> GeneratePipeline(MLContext context, ImageClassificationTrainer.Options options, string predictedLabelColumnName)
        {
            return context.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue(predictedLabelColumnName));
        }
    }
}
