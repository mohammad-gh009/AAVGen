import json
import matplotlib.pyplot as plt
import pandas as pd
# import torch
# from datasets import load_from_disk , Dataset
# from torch.utils.data import DataLoader
import json
# from transformers import AutoTokenizer,AutoModel, Trainer, TrainingArguments , EsmForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np
import seaborn as sns
from scipy import stats
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calibration_plot(y_pred, y_true, filename='calibration.png'):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    n_bins = 50
    df['bin'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop')
    calibration = df.groupby('bin').agg({'y_pred': 'mean', 'y_true': 'mean'}).reset_index()
    calibration['bin_mid'] = calibration['bin'].apply(lambda x: (x.left + x.right) / 2)

    plt.figure(figsize=(8, 6))
    plt.plot(calibration['bin_mid'], calibration['y_true'], marker='o', linestyle='-', label='Calibration Line')
    plt.plot([df['y_true'].min(), df['y_true'].max()],
             [df['y_true'].min(), df['y_true'].max()], 'r--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Value (Bin Midpoint)')
    plt.ylabel('Mean Actual Value')
    plt.title('Calibration Plot (Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


    

    


def calculate_correlation_stats(y_pred, y_true, alpha=0.05):
    """
    Calculate various correlation statistics (r values) between predicted and true values.
    
    Parameters:
    y_pred: array-like, predicted values
    y_true: array-like, true values
    alpha: float, significance level for statistical tests (default: 0.05)
    
    Returns:
    dict: Dictionary containing all correlation statistics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    print("="*60)
    print("CORRELATION STATISTICS (r values)")
    print("="*60)
    
    # 1. Pearson correlation coefficient (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    print(f"Pearson correlation (r): {pearson_r:.4f}")
    print(f"  p-value: {pearson_p:.6f}")
    print(f"  Significance: {'Significant' if pearson_p < alpha else 'Not significant'}")
    print(f"  Interpretation: {'Strong' if abs(pearson_r) > 0.7 else 'Moderate' if abs(pearson_r) > 0.3 else 'Weak'} {'positive' if pearson_r > 0 else 'negative'} linear correlation")
    
    # 2. Spearman correlation coefficient (monotonic relationship)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    print(f"\nSpearman correlation (ρ): {spearman_r:.4f}")
    print(f"  p-value: {spearman_p:.6f}")
    print(f"  Significance: {'Significant' if spearman_p < alpha else 'Not significant'}")
    print(f"  Interpretation: {'Strong' if abs(spearman_r) > 0.7 else 'Moderate' if abs(spearman_r) > 0.3 else 'Weak'} {'positive' if spearman_r > 0 else 'negative'} monotonic correlation")
    
    # 3. Kendall's tau (another rank correlation)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    print(f"\nKendall's tau (τ): {kendall_tau:.4f}")
    print(f"  p-value: {kendall_p:.6f}")
    print(f"  Significance: {'Significant' if kendall_p < alpha else 'Not significant'}")
    print(f"  Interpretation: {'Strong' if abs(kendall_tau) > 0.7 else 'Moderate' if abs(kendall_tau) > 0.3 else 'Weak'} {'positive' if kendall_tau > 0 else 'negative'} rank correlation")
    
    # 4. R-squared (coefficient of determination)
    from sklearn.metrics import r2_score
    r_squared = r2_score(y_true, y_pred)
    print(f"\nR-squared (R²): {r_squared:.4f}")
    print(f"  Interpretation: {r_squared*100:.1f}% of variance in true values is explained by predicted values")
    
    # 5. Adjusted R-squared (if you have multiple features, assume 1 feature for now)
    n_features = 1  # You can modify this based on your model
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_features - 1)
    print(f"Adjusted R-squared: {adj_r_squared:.4f}")
    
    # 6. Coefficient of correlation (same as Pearson but with confidence interval)
    # Calculate confidence interval for Pearson correlation
    def pearson_confidence_interval(r, n, confidence=0.95):
        """Calculate confidence interval for Pearson correlation coefficient"""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Fisher z-transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        z_lower = z_r - z_score * se
        z_upper = z_r + z_score * se
        
        # Transform back to correlation scale
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return r_lower, r_upper
    
    ci_lower, ci_upper = pearson_confidence_interval(pearson_r, n)
    print(f"\nPearson r 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # 7. Effect size interpretation
    print(f"\n" + "="*60)
    print("EFFECT SIZE INTERPRETATIONS")
    print("="*60)
    
    def interpret_correlation(r):
        abs_r = abs(r)
        if abs_r >= 0.9:
            return "Very strong"
        elif abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.5:
            return "Moderate"
        elif abs_r >= 0.3:
            return "Small to moderate"
        elif abs_r >= 0.1:
            return "Small"
        else:
            return "Negligible"
    
    print(f"Pearson r effect size: {interpret_correlation(pearson_r)}")
    print(f"Spearman ρ effect size: {interpret_correlation(spearman_r)}")
    print(f"Kendall τ effect size: {interpret_correlation(kendall_tau)}")
    
    # 8. Additional correlation-related statistics
    print(f"\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    
    # Root mean square correlation
    rms_correlation = np.sqrt(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))))
    print(f"RMS Correlation: {rms_correlation:.4f}")
    
    # Concordance correlation coefficient (for agreement)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    concordance_cc = (2 * pearson_r * np.sqrt(var_true) * np.sqrt(var_pred)) / \
                    (var_true + var_pred + (mean_true - mean_pred)**2)
    print(f"Concordance Correlation Coefficient: {concordance_cc:.4f}")
    
    # 9. Linearity Tests
    print(f"\n" + "="*60)
    print("LINEARITY TESTS")
    print("="*60)
    
    # Rainbow test for linearity (simplified version using polynomial fit)
    def rainbow_linearity_test(x, y, degree=2):
        """Simplified Rainbow test for linearity"""
        try:
            # Fit linear model
            linear_coef = np.polyfit(x, y, 1)
            linear_pred = np.polyval(linear_coef, x)
            ssr_linear = np.sum((y - linear_pred)**2)
            
            # Fit polynomial model
            poly_coef = np.polyfit(x, y, degree)
            poly_pred = np.polyval(poly_coef, x)
            ssr_poly = np.sum((y - poly_pred)**2)
            
            # F-test for nested models
            f_stat = ((ssr_linear - ssr_poly) / (degree - 1)) / (ssr_poly / (n - degree - 1))
            f_p = 1 - stats.f.cdf(f_stat, degree - 1, n - degree - 1)
            
            return f_stat, f_p
        except:
            return None, None
    
    rainbow_f, rainbow_p = rainbow_linearity_test(y_pred, y_true)
    if rainbow_f is not None:
        print(f"Rainbow Linearity Test:")
        print(f"  F-statistic: {rainbow_f:.4f}")
        print(f"  p-value: {rainbow_p:.6f}")
        print(f"  Result: {'Linear relationship' if rainbow_p > alpha else 'Non-linear relationship detected'}")
    
    # 10. Homoscedasticity Tests
    print(f"\n" + "="*60)
    print("HOMOSCEDASTICITY TESTS")
    print("="*60)
    
    residuals = y_true - y_pred
    
    # Breusch-Pagan test (simplified)
    def breusch_pagan_test(residuals, fitted):
        """Simplified Breusch-Pagan test for homoscedasticity"""
        try:
            squared_residuals = residuals**2
            # Regression of squared residuals on fitted values
            correlation = np.corrcoef(fitted, squared_residuals)[0, 1]
            
            # Convert to test statistic (approximate)
            n_obs = len(residuals)
            bp_stat = n_obs * correlation**2
            bp_p = 1 - stats.chi2.cdf(bp_stat, 1)
            
            return bp_stat, bp_p, correlation
        except:
            return None, None, None
    
    bp_stat, bp_p, bp_corr = breusch_pagan_test(residuals, y_pred)
    if bp_stat is not None:
        print(f"Breusch-Pagan Test:")
        print(f"  Test statistic: {bp_stat:.4f}")
        print(f"  p-value: {bp_p:.6f}")
        print(f"  Correlation (fitted vs residuals²): {bp_corr:.4f}")
        print(f"  Result: {'Homoscedastic' if bp_p > alpha else 'Heteroscedastic'}")
    
    # Goldfeld-Quandt test (divide data into groups)
    def goldfeld_quandt_test(residuals, fitted):
        """Goldfeld-Quandt test for homoscedasticity"""
        try:
            n_obs = len(residuals)
            # Sort by fitted values
            sorted_indices = np.argsort(fitted)
            sorted_residuals = residuals[sorted_indices]
            
            # Split into first and last third
            split_size = n_obs // 3
            first_third = sorted_residuals[:split_size]
            last_third = sorted_residuals[-split_size:]
            
            # Calculate variances
            var1 = np.var(first_third, ddof=1)
            var2 = np.var(last_third, ddof=1)
            
            # F-test for equal variances
            f_stat = max(var1, var2) / min(var1, var2)
            f_p = 2 * (1 - stats.f.cdf(f_stat, split_size-1, split_size-1))
            
            return f_stat, f_p
        except:
            return None, None
    
    gq_stat, gq_p = goldfeld_quandt_test(residuals, y_pred)
    if gq_stat is not None:
        print(f"\nGoldfeld-Quandt Test:")
        print(f"  F-statistic: {gq_stat:.4f}")
        print(f"  p-value: {gq_p:.6f}")
        print(f"  Result: {'Homoscedastic' if gq_p > alpha else 'Heteroscedastic'}")
    
    # 11. Independence Tests
    print(f"\n" + "="*60)
    print("INDEPENDENCE TESTS")
    print("="*60)
    
    # Ljung-Box test for autocorrelation in residuals
    def ljung_box_test(residuals, lags=10):
        """Ljung-Box test for autocorrelation"""
        try:
            n_obs = len(residuals)
            if lags >= n_obs:
                lags = min(10, n_obs // 4)
            
            # Calculate autocorrelations
            autocorrs = []
            for lag in range(1, lags + 1):
                if lag < n_obs:
                    ac = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    if not np.isnan(ac):
                        autocorrs.append(ac**2)
            
            if autocorrs:
                # Ljung-Box statistic
                lb_stat = n_obs * (n_obs + 2) * sum([(ac / (n_obs - i - 1)) for i, ac in enumerate(autocorrs)])
                lb_p = 1 - stats.chi2.cdf(lb_stat, len(autocorrs))
                return lb_stat, lb_p
            else:
                return None, None
        except:
            return None, None
    
    lb_stat, lb_p = ljung_box_test(residuals)
    if lb_stat is not None:
        print(f"Ljung-Box Test (Autocorrelation):")
        print(f"  Test statistic: {lb_stat:.4f}")
        print(f"  p-value: {lb_p:.6f}")
        print(f"  Result: {'No autocorrelation' if lb_p > alpha else 'Autocorrelation detected'}")
    
    # Runs test for randomness
    def runs_test(residuals):
        """Wald-Wolfowitz runs test for randomness"""
        try:
            # Convert residuals to binary (above/below median)
            median_res = np.median(residuals)
            binary = (residuals > median_res).astype(int)
            
            # Count runs
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i-1]:
                    runs += 1
            
            # Expected runs and variance
            n1 = np.sum(binary == 1)
            n2 = np.sum(binary == 0)
            
            if n1 > 0 and n2 > 0:
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
                
                # Z-test
                z_stat = (runs - expected_runs) / np.sqrt(var_runs)
                runs_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                return runs, expected_runs, z_stat, runs_p
            else:
                return None, None, None, None
        except:
            return None, None, None, None
    
    runs, exp_runs, runs_z, runs_p = runs_test(residuals)
    if runs is not None:
        print(f"\nWald-Wolfowitz Runs Test:")
        print(f"  Observed runs: {runs}")
        print(f"  Expected runs: {exp_runs:.2f}")
        print(f"  Z-statistic: {runs_z:.4f}")
        print(f"  p-value: {runs_p:.6f}")
        print(f"  Result: {'Random' if runs_p > alpha else 'Non-random pattern detected'}")
    
    # 12. Outlier Tests
    print(f"\n" + "="*60)
    print("OUTLIER DETECTION TESTS")
    print("="*60)
    
    # Modified Z-score for outliers
    def modified_z_score(data, threshold=3.5):
        """Modified Z-score using median absolute deviation"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = np.mean(np.abs(data - median))
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        return outliers, modified_z_scores
    
    outliers_res, mod_z_scores = modified_z_score(residuals)
    n_outliers = np.sum(outliers_res)
    outlier_percentage = (n_outliers / n) * 100
    
    print(f"Modified Z-Score Outlier Detection:")
    print(f"  Number of outliers: {n_outliers} ({outlier_percentage:.1f}%)")
    print(f"  Outlier threshold: 3.5")
    
    # Cook's Distance (simplified)
    def cooks_distance(residuals, fitted_values):
        """Calculate Cook's distance"""
        try:
            # Standardized residuals
            mse = np.mean(residuals**2)
            standardized_res = residuals / np.sqrt(mse)
            
            # Leverage (simplified - assumes equal leverage)
            n_params = 2  # intercept + slope
            leverage = n_params / n
            
            # Cook's distance
            cooks_d = (standardized_res**2 / n_params) * (leverage / (1 - leverage))
            
            # Threshold: 4/n
            threshold = 4 / n
            influential = cooks_d > threshold
            
            return cooks_d, influential, threshold
        except:
            return None, None, None
    
    cooks_d, influential, cook_threshold = cooks_distance(residuals, y_pred)
    if cooks_d is not None:
        n_influential = np.sum(influential)
        print(f"\nCook's Distance:")
        print(f"  Influential observations: {n_influential} ({(n_influential/n)*100:.1f}%)")
        print(f"  Threshold (4/n): {cook_threshold:.6f}")
        print(f"  Max Cook's D: {np.max(cooks_d):.6f}")
    
    # 13. Distribution Tests
    print(f"\n" + "="*60)
    print("DISTRIBUTION TESTS")
    print("="*60)
    
    # Anderson-Darling test for normality
    try:
        ad_stat, ad_critical, ad_significance = stats.anderson(residuals, dist='norm')
        print(f"Anderson-Darling Normality Test:")
        print(f"  Test statistic: {ad_stat:.4f}")
        print(f"  Critical values: {ad_critical}")
        print(f"  Significance levels: {ad_significance}%")
        
        # Check at 5% significance level
        critical_5pct = ad_critical[2] if len(ad_critical) > 2 else ad_critical[-1]
        print(f"  Result at 5%: {'Normal distribution' if ad_stat < critical_5pct else 'Not normal distribution'}")
    except:
        print("Anderson-Darling test could not be computed")
    
    # D'Agostino's normality test
    try:
        dag_stat, dag_p = stats.normaltest(residuals)
        print(f"\nD'Agostino's Normality Test:")
        print(f"  Test statistic: {dag_stat:.4f}")
        print(f"  p-value: {dag_p:.6f}")
        print(f"  Result: {'Normal distribution' if dag_p > alpha else 'Not normal distribution'}")
    except:
        print("\nD'Agostino's test could not be computed")
    
    # Return all results
    results = {
        'correlations': {
            'pearson_r': pearson_r,
            'pearson_p_value': pearson_p,
            'pearson_ci_lower': ci_lower,
            'pearson_ci_upper': ci_upper,
            'spearman_r': spearman_r,
            'spearman_p_value': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p_value': kendall_p,
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'concordance_cc': concordance_cc,
            'rms_correlation': rms_correlation,
        },
        'linearity_tests': {
            'rainbow_f_stat': rainbow_f,
            'rainbow_p_value': rainbow_p
        },
        'homoscedasticity_tests': {
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_p': bp_p,
            'bp_correlation': bp_corr,
            'goldfeld_quandt_stat': gq_stat,
            'goldfeld_quandt_p': gq_p
        },
        'independence_tests': {
            'ljung_box_stat': lb_stat,
            'ljung_box_p': lb_p,
            'runs_test_observed': runs,
            'runs_test_expected': exp_runs,
            'runs_test_z': runs_z,
            'runs_test_p': runs_p
        },
        'outlier_tests': {
            'n_outliers_modified_z': n_outliers,
            'outlier_percentage': outlier_percentage,
            'n_influential_cooks': n_influential if influential is not None else None,
            'max_cooks_distance': np.max(cooks_d) if cooks_d is not None else None
        },
        'normality_tests': {
            'anderson_darling_stat': ad_stat if 'ad_stat' in locals() else None,
            'dagostino_stat': dag_stat if 'dag_stat' in locals() else None,
            'dagostino_p': dag_p if 'dag_p' in locals() else None
        },
        'effect_sizes': {
            'pearson': interpret_correlation(pearson_r),
            'spearman': interpret_correlation(spearman_r),
            'kendall': interpret_correlation(kendall_tau)
        }
    }
    
    return results


def evaluate_model(model_path, tok_path, test_data_path):
    """
    Evaluate a pre-trained model on a test dataset.

    Args:
    - model_path (str): Path to the pre-trained model.
    - test_data (Dataset): Test dataset to evaluate the model on.
    - batch_size (int): Batch size for processing the test data.

    Returns:
    - y_true (list): List of true labels.
    - y_pred (list): List of predicted labels.
    """
    # Load pre-trained model and tokenizer
    model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    tokenized = load_from_disk(test_data_path)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataloader = DataLoader(tokenized, batch_size=64, collate_fn=data_collator)
    model.to(device)
    model.eval() 
    
    with torch.no_grad():
        all_preds = []
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits        
            y_pred = outputs.logits.squeeze().tolist()
            all_preds += y_pred

    y_true = [tensor.item() for tensor in tokenized["label"]]

    return y_true,all_preds

def calculate_metrics(y_pred, y_true , alpha = 0.05):
    """
    Calculate regression metrics and perform statistical tests.
    
    Parameters:
    y_true: array-like, true values
    y_pred: array-like, predicted values
    alpha: float, significance level for statistical tests (default: 0.05)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    # Basic regression metrics
    val_rmse = mean_squared_error(y_true, y_pred)#, squared=False
    val_mae = mean_absolute_error(y_true, y_pred)
    val_r2 = r2_score(y_true, y_pred)
    val_mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Pearson correlation coefficient
    val_pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Pseudo-Huber loss (with delta=1)
    delta = 1.0
    residuals = y_true - y_pred
    val_pseudo_huber = np.mean(delta**2 * (np.sqrt(1 + (residuals/delta)**2) - 1))
    
    print("="*60)
    print("REGRESSION METRICS")
    print("="*60)
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE: {val_mae:.4f}")
    print(f"R²: {val_r2:.4f}")
    print(f"MAPE: {val_mape:.4f}")
    print(f"Pearson Correlation: {val_pearson_corr:.4f} (p-value: {pearson_p:.6f})")
    print(f"Pseudo-Huber Loss: {val_pseudo_huber:.4f}")
    
    # Statistical Tests
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # 1. Normality test of residuals (Shapiro-Wilk)
    if n <= 5000:  # Shapiro-Wilk is reliable for n <= 5000
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk Normality Test:")
        print(f"  Statistic: {shapiro_stat:.4f}")
        print(f"  p-value: {shapiro_p:.6f}")
        print(f"  Result: {'Residuals are normally distributed' if shapiro_p > alpha else 'Residuals are NOT normally distributed'}")
    else:
        # Use Kolmogorov-Smirnov test for larger samples
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        print(f"Kolmogorov-Smirnov Normality Test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  p-value: {ks_p:.6f}")
        print(f"  Result: {'Residuals are normally distributed' if ks_p > alpha else 'Residuals are NOT normally distributed'}")
    
    # 2. Zero-mean test (One-sample t-test)
    t_stat, t_p = stats.ttest_1samp(residuals, 0)
    print(f"\nZero-Mean Test (One-sample t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_p:.6f}")
    print(f"  Result: {'Residuals have zero mean' if t_p > alpha else 'Residuals do NOT have zero mean (bias detected)'}")
    
    # 3. Durbin-Watson test for autocorrelation
    def durbin_watson(residuals):
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
    
    dw_stat = durbin_watson(residuals)
    print(f"\nDurbin-Watson Test (Autocorrelation):")
    print(f"  Statistic: {dw_stat:.4f}")
    print(f"  Interpretation: ", end="")
    if dw_stat < 1.5:
        print("Positive autocorrelation detected")
    elif dw_stat > 2.5:
        print("Negative autocorrelation detected")
    else:
        print("No significant autocorrelation")
    
    # 4. Jarque-Bera test (alternative normality test)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    print(f"\nJarque-Bera Normality Test:")
    print(f"  Statistic: {jb_stat:.4f}")
    print(f"  p-value: {jb_p:.6f}")
    print(f"  Result: {'Residuals are normally distributed' if jb_p > alpha else 'Residuals are NOT normally distributed'}")
    
    # 5. Breusch-Pagan test for heteroscedasticity (simplified version)
    # Regress squared residuals on predicted values
    try:
        squared_residuals = residuals**2
        correlation_coef = np.corrcoef(y_pred, squared_residuals)[0, 1]
        
        # Simple test based on correlation
        print(f"\nHomoscedasticity Assessment:")
        print(f"  Correlation(y_pred, residuals²): {correlation_coef:.4f}")
        if abs(correlation_coef) > 0.3:
            print(f"  Result: Heteroscedasticity detected (correlation > 0.3)")
        else:
            print(f"  Result: Homoscedasticity (no strong pattern in residuals)")
    except:
        print(f"\nHomoscedasticity Assessment: Could not compute")
    
    # 6. Model significance test (F-test for R²)
    if val_r2 > 0:
        f_stat = (val_r2 / (1 - val_r2)) * (n - 2)
        f_p = 1 - stats.f.cdf(f_stat, 1, n - 2)
        print(f"\nModel Significance Test (F-test for R²):")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {f_p:.6f}")
        print(f"  Result: {'Model is statistically significant' if f_p < alpha else 'Model is NOT statistically significant'}")
    
    # 7. Additional descriptive statistics
    print(f"\n" + "="*60)
    print("RESIDUAL STATISTICS")
    print("="*60)
    print(f"Mean of residuals: {np.mean(residuals):.6f}")
    print(f"Std of residuals: {np.std(residuals):.4f}")
    print(f"Min residual: {np.min(residuals):.4f}")
    print(f"Max residual: {np.max(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    # Return all metrics and test results as a dictionary
    results = {
        'metrics': {
            'rmse': val_rmse,
            'mae': val_mae,
            'r2': val_r2,
            'mape': val_mape,
            'pearson_corr': val_pearson_corr,
            'pearson_p_value': pearson_p,
            'pseudo_huber': val_pseudo_huber
        },
        'residuals_stats': {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        },
        'statistical_tests': {
            'zero_mean_test_p': t_p,
            'jarque_bera_p': jb_p,
            'durbin_watson_stat': dw_stat,
            'heteroscedasticity_corr': correlation_coef if 'correlation_coef' in locals() else None
        }
    }
    
    return results

def plot_training_loss(file_path, out_path , n_val_step):
    """
    Generate a plot of training and evaluation loss from a JSON file.

    Args:
    file_path (str): Path to the JSON file containing the training log history.
    n_val_step: the validation step that is used in the training
    Returns:
    None
    """

    # Open and load the JSON file
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    # Extract log history from the JSON data
    data = json_data["log_history"]

    # Extract steps and losses from the data
    steps = [entry["step"] for entry in data if "step" in entry and "loss" in entry]
    steps_ev = [entry["step"] for entry in data if "step" in entry and "eval_loss" in entry]
    losses_t = [entry["loss"] for entry in data if "step" in entry and "loss" in entry]
    losses_e = [entry["eval_loss"] for entry in data if "step" in entry and "eval_loss" in entry]

    # Filter to show every 100 steps
    filtered_steps = []
    filtered_losses_t = []
    for i, step in enumerate(steps):
        if step % n_val_step == 0:
            filtered_steps.append(step)
            filtered_losses_t.append(losses_t[i])

    # Create a line plot
    plt.plot(filtered_steps, filtered_losses_t, marker="o", label="Training Loss")
    plt.plot(steps_ev, losses_e, marker="o", label="Evaluation Loss")
    # plt.ylim(0 , 2)

    # Set labels and title
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss ")
    plt.legend()

    # Save the plot to a file
    plt.savefig( out_path , bbox_inches="tight")