import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

class AirQualityPredictor:
    """Predict future air quality based on current conditions"""

    def __init__(self, decay_rate=0.25):
        """
        Initialize predictor

        Parameters:
        -----------
        decay_rate : float - PM2.5 decay rate (default 0.25 for 4-day half-life)
        """
        self.decay_rate = decay_rate
        self.forecast_history = []

    def generate_forecast(self, current_pm25, wind_speed, humidity, 
                         temperature=25, days=7, include_confidence=True):
        """
        Generate PM2.5 and AQI forecast for next N days

        Uses exponential decay with meteorological adjustments
        and stochastic noise for realistic predictions

        Parameters:
        -----------
        current_pm25 : float - Current PM2.5 concentration
        wind_speed : float - Average wind speed (km/h)
        humidity : float - Average humidity (%)
        temperature : float - Average temperature (°C)
        days : int - Number of days to forecast
        include_confidence : bool - Include confidence intervals

        Returns:
        --------
        list : Forecast data for each day
        """
        forecast = []

        for i in range(days):
            # Exponential decay (PM2.5 reduces over time)
            decay = math.exp(-i * self.decay_rate)

            # Weather impact factor
            wind_dispersion = max(0.5, 1 - (wind_speed / 100))
            humidity_dampening = max(0.6, 1 - (humidity / 150))
            temp_factor = 1 + ((temperature - 25) / 100)

            weather_impact = wind_dispersion * humidity_dampening * temp_factor

            # Base prediction with decay and weather
            base_pm25 = current_pm25 * decay * weather_impact

            # Add realistic stochastic variation (±15%)
            noise = (np.random.random() - 0.5) * base_pm25 * 0.3

            # Final PM2.5 prediction (minimum 5 μg/m³)
            pm25 = max(5, base_pm25 + noise)

            # Calculate AQI from PM2.5
            aqi = self.calculate_aqi(pm25)
            category = self.get_aqi_category(aqi)

            # Build forecast entry
            forecast_entry = {
                'day': i + 1,
                'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'pm25': round(pm25, 2),
                'aqi': aqi,
                'category': category,
                'confidence': max(50, 100 - (i * 7))  # Decreases with time
            }

            # Add confidence intervals if requested
            if include_confidence:
                margin = pm25 * (0.15 + i * 0.05)  # Wider with time
                forecast_entry['pm25_lower'] = round(max(0, pm25 - margin), 2)
                forecast_entry['pm25_upper'] = round(pm25 + margin, 2)
                forecast_entry['aqi_lower'] = self.calculate_aqi(forecast_entry['pm25_lower'])
                forecast_entry['aqi_upper'] = self.calculate_aqi(forecast_entry['pm25_upper'])

            forecast.append(forecast_entry)

        self.forecast_history.append({
            'timestamp': datetime.now(),
            'initial_pm25': current_pm25,
            'forecast': forecast
        })

        return forecast

    def calculate_aqi(self, pm25):
        """Calculate AQI from PM2.5 using EPA formula"""
        if pm25 <= 12.0:
            return round((50 / 12.0) * pm25)
        elif pm25 <= 35.4:
            return round(50 + ((100 - 50) / (35.4 - 12.0)) * (pm25 - 12.0))
        elif pm25 <= 55.4:
            return round(100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4))
        elif pm25 <= 150.4:
            return round(150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4))
        elif pm25 <= 250.4:
            return round(200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4))
        else:
            return round(300 + ((500 - 300) / (500.4 - 250.4)) * min(pm25 - 250.4, 250))

    def get_aqi_category(self, aqi):
        """Get AQI category from value"""
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    def predict_peak_day(self, forecast):
        """Identify the day with worst predicted air quality"""
        peak = max(forecast, key=lambda x: x['aqi'])
        return peak

    def predict_safe_day(self, forecast, threshold_aqi=100):
        """Find first day when AQI drops below threshold"""
        for day in forecast:
            if day['aqi'] < threshold_aqi:
                return day
        return None

    def generate_alerts(self, forecast, threshold_aqi=100):
        """
        Generate alerts for days exceeding AQI threshold

        Parameters:
        -----------
        forecast : list - Forecast data
        threshold_aqi : int - AQI threshold for alerts (default 100)

        Returns:
        --------
        list : Alert information for high AQI days
        """
        alerts = []
        for day in forecast:
            if day['aqi'] > threshold_aqi:
                alert_level = (
                    'CRITICAL' if day['aqi'] > 200 else
                    'HIGH' if day['aqi'] > 150 else
                    'MODERATE'
                )
                alerts.append({
                    'day': day['day'],
                    'date': day['date'],
                    'aqi': day['aqi'],
                    'category': day['category'],
                    'alert_level': alert_level,
                    'pm25': day['pm25']
                })
        return alerts

    def export_forecast(self, forecast, filename='air_quality_forecast.csv'):
        """Export forecast to CSV"""
        df = pd.DataFrame(forecast)
        df.to_csv(filename, index=False)
        print(f"✅ Forecast exported to {filename}")
        return df

    def print_forecast(self, forecast):
        """Print formatted forecast report"""
        print("\n" + "="*70)
        print("AIR QUALITY FORECAST REPORT")
        print("="*70)
        print(f"Forecast Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Forecast Period: {len(forecast)} days")
        print("\n{:<6} {:<12} {:<10} {:<6} {:<30} {:<10}".format(
            "Day", "Date", "PM2.5", "AQI", "Category", "Confidence"
        ))
        print("-" * 70)

        for day in forecast:
            confidence_bar = "█" * (day['confidence'] // 10)
            print("{:<6} {:<12} {:<10.2f} {:<6} {:<30} {:<10}".format(
                day['day'],
                day['date'],
                day['pm25'],
                day['aqi'],
                day['category'],
                f"{confidence_bar} {day['confidence']}%"
            ))

        print("="*70)

        # Summary statistics
        avg_aqi = np.mean([d['aqi'] for d in forecast])
        max_aqi = max([d['aqi'] for d in forecast])
        min_aqi = min([d['aqi'] for d in forecast])

        print("\nFORECAST SUMMARY:")
        print(f"  Average AQI: {avg_aqi:.1f}")
        print(f"  Maximum AQI: {max_aqi} (Day {[d['day'] for d in forecast if d['aqi'] == max_aqi][0]})")
        print(f"  Minimum AQI: {min_aqi} (Day {[d['day'] for d in forecast if d['aqi'] == min_aqi][0]})")
        print("="*70 + "\n")


class WildfireRiskPredictor:
    """Predict wildfire risk based on meteorological conditions"""

    def __init__(self):
        self.risk_history = []

    def predict_fire_probability(self, temp, humidity, wind, rain, 
                                 vegetation_dryness=0.7, drought_severity=0.5):
        """
        Predict probability of wildfire occurrence

        Uses logistic regression model calibrated to historical wildfire data

        Parameters:
        -----------
        temp : float - Temperature (°C)
        humidity : float - Relative humidity (%)
        wind : float - Wind speed (km/h)
        rain : float - Recent rainfall (mm)
        vegetation_dryness : float - Vegetation moisture (0-1, 1=very dry)
        drought_severity : float - Long-term drought index (0-1)

        Returns:
        --------
        float : Fire probability (0-100%)
        """
        # Logistic regression with environmental factors
        z = (-5.2 + 
             (0.08 * temp) -              # Higher temp increases risk
             (0.05 * humidity) +          # Higher humidity decreases risk
             (0.04 * wind) -              # Higher wind increases spread
             (0.5 * rain) +               # Recent rain reduces risk
             (3.0 * vegetation_dryness) + # Dry vegetation increases risk
             (2.0 * drought_severity))    # Drought increases risk

        # Sigmoid function for probability
        probability = 1 / (1 + math.exp(-z))

        # Record prediction
        self.risk_history.append({
            'timestamp': datetime.now(),
            'probability': probability * 100,
            'temp': temp,
            'humidity': humidity,
            'wind': wind,
            'rain': rain
        })

        return min(100, max(0, probability * 100))

    def assess_risk_factors(self, temp, humidity, wind, rain):
        """
        Assess individual risk factors

        Returns risk level (LOW/MODERATE/HIGH) for each factor
        """
        risk_factors = {
            'temperature': 'HIGH' if temp > 35 else 'MODERATE' if temp > 30 else 'LOW',
            'humidity': 'HIGH' if humidity < 20 else 'MODERATE' if humidity < 40 else 'LOW',
            'wind': 'HIGH' if wind > 40 else 'MODERATE' if wind > 25 else 'LOW',
            'drought': 'HIGH' if rain < 1 else 'MODERATE' if rain < 5 else 'LOW'
        }

        # Calculate overall risk score
        score_map = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
        total_score = sum(score_map[level] for level in risk_factors.values())
        max_score = 4 * 3  # 4 factors × max score of 3

        overall_risk = (
            'EXTREME' if total_score >= 11 else
            'HIGH' if total_score >= 9 else
            'MODERATE' if total_score >= 6 else
            'LOW'
        )

        risk_factors['overall'] = overall_risk
        risk_factors['score'] = total_score
        risk_factors['max_score'] = max_score

        return risk_factors

    def generate_risk_report(self, conditions):
        """
        Generate comprehensive wildfire risk report

        Parameters:
        -----------
        conditions : dict with keys temperature, humidity, wind_speed, rainfall

        Returns:
        --------
        dict : Complete risk assessment
        """
        probability = self.predict_fire_probability(
            conditions['temperature'],
            conditions['humidity'],
            conditions['wind_speed'],
            conditions['rainfall']
        )

        risk_factors = self.assess_risk_factors(
            conditions['temperature'],
            conditions['humidity'],
            conditions['wind_speed'],
            conditions['rainfall']
        )

        # Risk level based on probability
        risk_level = (
            'EXTREME' if probability > 85 else
            'HIGH' if probability > 70 else
            'MODERATE' if probability > 50 else
            'LOW'
        )

        # Recommended actions
        actions = self.get_recommended_actions(risk_level)

        return {
            'probability': round(probability, 2),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_actions': actions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_recommended_actions(self, risk_level):
        """Get recommended actions based on risk level"""
        actions = {
            'EXTREME': [
                'Immediate evacuation may be necessary',
                'No outdoor burning or spark-producing activities',
                'Emergency services on high alert',
                'Community notification systems activated'
            ],
            'HIGH': [
                'Prepare for possible evacuation',
                'Avoid all outdoor burning',
                'Monitor weather and fire updates',
                'Have emergency supplies ready'
            ],
            'MODERATE': [
                'Exercise caution with fire use',
                'Follow local fire restrictions',
                'Be aware of changing conditions',
                'Keep informed of fire weather warnings'
            ],
            'LOW': [
                'Normal precautions apply',
                'Follow standard fire safety practices',
                'Monitor conditions periodically'
            ]
        }
        return actions.get(risk_level, actions['MODERATE'])
        def print_risk_report(self, report): """Print formatted risk report"""
        print("\n" + "="*70)
        print("WILDFIRE RISK ASSESSMENT REPORT")
        print("="*70)
        print(f"Assessment Time: {report['timestamp']}")
        print(f"\nFire Probability: {report['probability']:.1f}%")
        print(f"Risk Level: {report['risk_level']}")

        print("\n--- RISK FACTORS ---")
        factors = report['risk_factors']
        print(f"Temperature Risk: {factors['temperature']}")
        print(f"Humidity Risk: {factors['humidity']}")
        print(f"Wind Risk: {factors['wind']}")
        print(f"Drought Risk: {factors['drought']}")
        print(f"Overall Score: {factors['score']}/{factors['max_score']}")

        print("\n--- RECOMMENDED ACTIONS ---")
        for i, action in enumerate(report['recommended_actions'], 1):
            print(f"{i}. {action}")

        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    print("WILDFIRE AIR QUALITY PREDICTION SYSTEM")
    print("="*70)

    # Example 1: Air Quality Forecast
    print("\nEXAMPLE 1: 7-Day Air Quality Forecast")
    print("-" * 70)

    predictor = AirQualityPredictor(decay_rate=0.25)

    # Current conditions
    current_pm25 = 250.5  # High PM2.5 from active wildfire
    wind_speed = 25
    humidity = 20
    temperature = 35

    # Generate forecast
    forecast = predictor.generate_forecast(
        current_pm25=current_pm25,
        wind_speed=wind_speed,
        humidity=humidity,
        temperature=temperature,
        days=7,
        include_confidence=True
    )

    predictor.print_forecast(forecast)

    # Find peak pollution day
    peak_day = predictor.predict_peak_day(forecast)
    print(f"⚠️  Peak pollution expected: Day {peak_day['day']} ({peak_day['date']})")
    print(f"   Predicted AQI: {peak_day['aqi']} ({peak_day['category']})")

    # Find when air quality improves
    safe_day = predictor.predict_safe_day(forecast, threshold_aqi=100)
    if safe_day:
        print(f"✅ Air quality improves: Day {safe_day['day']} ({safe_day['date']})")
        print(f"   Predicted AQI: {safe_day['aqi']} ({safe_day['category']})")
    else:
        print(f"⚠️  No improvement to moderate levels within forecast period")

    # Generate alerts
    alerts = predictor.generate_alerts(forecast, threshold_aqi=100)
    if alerts:
        print(f"\n🚨 {len(alerts)} alert(s) generated:")
        for alert in alerts:
            print(f"   Day {alert['day']} ({alert['date']}): {alert['alert_level']} - AQI {alert['aqi']}")

    # Export forecast
    predictor.export_forecast(forecast, 'forecast_7day.csv')

    # Example 2: Wildfire Risk Assessment
    print("\n\nEXAMPLE 2: Wildfire Risk Assessment")
    print("-" * 70)

    risk_predictor = WildfireRiskPredictor()

    conditions = {
        'temperature': 35,
        'humidity': 20,
        'wind_speed': 25,
        'rainfall': 0
    }

    risk_report = risk_predictor.generate_risk_report(conditions)
    risk_predictor.print_risk_report(risk_report)

    # Example 3: Multi-day risk forecast
    print("\nEXAMPLE 3: 5-Day Fire Risk Forecast")
    print("-" * 70)

    print("\n{:<6} {:<8} {:<10} {:<10} {:<15} {:<12}".format(
        "Day", "Temp(°C)", "Humid(%)", "Wind(km/h)", "Probability(%)", "Risk Level"
    ))
    print("-" * 70)

    # Simulate changing conditions
    for day in range(1, 6):
        temp = 35 - day * 1.5  # Gradually cooling
        humid = 20 + day * 5    # Humidity increasing
        wind = 25 - day * 2     # Wind decreasing
        rain = 0 if day < 3 else day * 2  # Rain starts day 3

        prob = risk_predictor.predict_fire_probability(temp, humid, wind, rain)
        level = 'HIGH' if prob > 70 else 'MODERATE' if prob > 50 else 'LOW'

        print("{:<6} {:<8.1f} {:<10.0f} {:<10.0f} {:<15.1f} {:<12}".format(
            day, temp, humid, wind, prob, level
        ))

    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("Files created: forecast_7day.csv")
    print("="*70)
