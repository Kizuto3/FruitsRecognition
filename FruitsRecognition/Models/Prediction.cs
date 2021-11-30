using Prism.Mvvm;
using System;
using System.Windows.Media;
using Brush = System.Windows.Media.Brush;

namespace FruitsRecognition.Models
{
    public class Prediction : BindableBase
    {
        private string _label;
        public double _percentage;

        public string Label
        {
            get => _label;
            set => SetProperty(ref _label, value, nameof(Label));
        }

        public double Percentage
        {
            get => Math.Round(_percentage * 100, 2);
            set
            {
                SetProperty(ref _percentage, value, nameof(Percentage));
                RaisePropertyChanged(nameof(Width));
                RaisePropertyChanged(nameof(RectColor));
            }
        }

        public int Width => 350 * (int)Percentage / 100;

        public Brush RectColor 
        {
            get 
            {
                var normalizedPercentage = 255 * Percentage / 100;
                return new SolidColorBrush(System.Windows.Media.Color.FromArgb(255, (byte)(255 - normalizedPercentage), 
                                                                               (byte)normalizedPercentage, 0));
            }
        }
    }
}
