using FruitsRecognition.MLContextFactory.Data;
using FruitsRecognition.Models;
using FruitsRecognition.ModelTrainer;
using FruitsRecognition.Predictor;
using Microsoft.Win32;
using Prism.Commands;
using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace FruitsRecognition.ViewModels
{
    public class MainWindowViewModel : BindableBase
    {
        private Trainer _trainer;
        private PredictorModel _predictor;
        private int _seed;

        private bool _isBusy;
        private string _busyStatement;
        private byte[] _currentImage;
        private string _predictionResult;

        public bool IsBusy
        {
            get => _isBusy;
            set => SetProperty(ref _isBusy, value, nameof(IsBusy));
        }

        public string BusyStatement
        {
            get => _busyStatement; 
            set => SetProperty(ref _busyStatement, value, nameof(BusyStatement));
        }

        public byte[] CurrentImage
        {
            get => _currentImage;
            set => SetProperty(ref _currentImage, value, nameof(CurrentImage));
        }

        public string PredictionResult
        {
            get => _predictionResult;
            set => SetProperty(ref _predictionResult, value, nameof(PredictionResult));
        }

        public ObservableCollection<Prediction> Predictions { get; set; }

        public DelegateCommand PredictCommand { get; private set; }
        public DelegateCommand TrainCommand { get; private set; }
        public DelegateCommand LoadModelCommand { get; private set; }

        public MainWindowViewModel()
        {
            _trainer = new Trainer();
            _predictor = new PredictorModel();
            _seed = 0;

            Predictions = new ObservableCollection<Prediction>();

            PredictCommand = new DelegateCommand(Predict);
            TrainCommand = new DelegateCommand(Train);
            LoadModelCommand = new DelegateCommand(LoadModel);
        }

        private async void LoadModel()
        {
            var dialog = new OpenFileDialog()
            {
                Filter = "Zip Archive (.zip)|*.zip|All Files (*.*)|(*.*)"
            };

            if (dialog.ShowDialog() == true)
            {
                IsBusy = true;
                BusyStatement = "Loading training model. Please wait...";

                await Task.Run(() =>
                {
                    _predictor.LoadModel(dialog.FileName, _seed);

                    Dispatcher.CurrentDispatcher.Invoke(() =>
                    {
                        IsBusy = false;
                        BusyStatement = "";
                    });
                });
            }
        }

        private async void Predict()
        {
            if(_predictor.Predictor == null)
            {
                MessageBox.Show("No model loaded!\nPlease load a trained model", "Error!");
                return;
            }

            var dialog = new OpenFileDialog();

            if (dialog.ShowDialog() == true)
            {
                Predictions.Clear();

                var image = new BitmapImage(new Uri(dialog.FileName));

                // Load the image into memory
                var encoder = new JpegBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(image));
                var input = new Input();

                using (var ms = new MemoryStream())
                {
                    encoder.Save(ms);
                    input.Image = ms.ToArray();
                    CurrentImage = input.Image;
                }

                IsBusy = true;
                BusyStatement = "Generating prediction. Please wait...";

                List<KeyValuePair<string, float>> res = null;

                await Task.Run(() =>
                {
                    res = _predictor.Predict(input);
                    int i = 0;

                    Dispatcher.CurrentDispatcher.Invoke(() =>
                    {
                        PredictionResult = $"Seems like this is a {res.First().Key}!";
                        IsBusy = false;
                        BusyStatement = "";
                    });

                    foreach (var value in res)
                    {
                        if (++i > 5)
                            break;

                        System.Windows.Application.Current.Dispatcher.BeginInvoke(() =>
                        {
                            Predictions.Add(new Prediction() { Label = value.Key, Percentage = value.Value });                           
                        });
                    }
                });
            }
        }

        private async void Train()
        {
            using var dialog = new FolderBrowserDialog();

            if (dialog.ShowDialog() == DialogResult.OK)
            {
                IsBusy = true;
                BusyStatement = "\t\tTraining the model.\nDepending on the data size this might take up to 5 minutes.\n\t\tPlease wait...";

                await Task.Run(() =>
                {
                    var model = _trainer.Train(dialog.SelectedPath, _seed);

                    Dispatcher.CurrentDispatcher.Invoke(() =>
                    {
                        IsBusy = false;
                        BusyStatement = "";

                        if (MessageBox.Show("Do you want to export the trained model?", "Traininig completed", MessageBoxButtons.YesNo) == DialogResult.Yes)
                        {
                            var saveDialog = new Microsoft.Win32.SaveFileDialog()
                            {
                                AddExtension = true,
                                DefaultExt = ".zip",
                                Title = "Save trained model",
                                Filter = "Zip Archive (*.zip)|*.zip",
                                FileName = "Trained_model"
                            };

                            if (saveDialog.ShowDialog() == true)
                            {
                                _trainer.ExportModel(model, saveDialog.FileName);
                            }
                        }
                    });
                });
            }
        }
    }
}
