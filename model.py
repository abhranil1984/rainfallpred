import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.output_linear(context)
        
        return output, attn_weights

class EnhancedFeatureAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super(EnhancedFeatureAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, features):
        energy = self.projection(features)
        weights = F.softmax(energy, dim=2)
        weighted_features = features * weights
        return weighted_features, weights

class EnhancedTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(EnhancedTemporalAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        outputs = (encoder_outputs * weights).sum(dim=1)
        return outputs, weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AdvancedAttentionNetwork(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size=128, num_heads=4, dropout=0.1):
        super(AdvancedAttentionNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=time_steps)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.feature_attention = EnhancedFeatureAttention(hidden_size, dropout)
        self.temporal_attention = EnhancedTemporalAttention(hidden_size, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        normed_embedding = self.layer_norm1(embedded)
        
        self_att_output, self_att_weights = self.self_attention(normed_embedding)
        self_att_output = self.dropout1(self_att_output) + normed_embedding
        self_att_output = self.layer_norm2(self_att_output)
        
        weighted_features, feature_weights = self.feature_attention(self_att_output)
        context, temporal_weights = self.temporal_attention(weighted_features)
        
        ff_output = self.feed_forward(context)
        ff_output = self.layer_norm3(ff_output + context)
        
        output = self.output_layer(ff_output)
        
        return output, (temporal_weights, feature_weights, self_att_weights)

class AdvancedAttentiveRFNetwork:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 input_size=None, time_steps=6, hidden_size=128, num_heads=4, dropout=0.1):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        )
        self.input_size = input_size
        self.time_steps = time_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.attention_net = AdvancedAttentionNetwork(
            input_size, 
            time_steps,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.attention_net.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.criterion = nn.MSELoss()
        
    def fit(self, X_seq, y, epochs=250, batch_size=32, validation_split=0.1):
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        
        dataset_size = len(X_tensor)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            TensorDataset(X_tensor, y_tensor), 
            [train_size, val_size]
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 150
        
        print("Training advanced attention network...")
        for epoch in range(epochs):
            self.attention_net.train()
            epoch_loss = 0
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                
                outputs, _ = self.attention_net(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.attention_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_X)
            
            train_loss = epoch_loss / train_size
            
            self.attention_net.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    outputs, _ = self.attention_net(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * len(batch_X)
            
            val_loss = val_loss / val_size
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.attention_net.eval()
        with torch.no_grad():
            _, (_, feature_weights, _) = self.attention_net(X_tensor)
            weighted_X_seq = X_seq * feature_weights.cpu().numpy()
            X_flat = weighted_X_seq.reshape(X_seq.shape[0], -1)
        
        print("Training Random Forest on attention-weighted features...")
        self.rf.fit(X_flat, y)
    
    def predict(self, X_seq):
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.attention_net.eval()
        with torch.no_grad():
            _, (temporal_weights, feature_weights, self_attn_weights) = self.attention_net(X_tensor)
            
            weighted_X_seq = X_seq * feature_weights.cpu().numpy()
            X_flat = weighted_X_seq.reshape(X_seq.shape[0], -1)
        
        y_pred = self.rf.predict(X_flat)
        
        attention_weights = (temporal_weights.cpu().numpy(), 
                           feature_weights.cpu().numpy(),
                           self_attn_weights.cpu().numpy())
        
        return y_pred, attention_weights

def save_fig(fig, filename, dpi=300, bbox_inches='tight'):
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

df = pd.read_csv('DATA.csv')

df_melted = pd.melt(df, id_vars=['PARAMETER', 'YEAR', 'REGION'], 
                    value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
                    var_name='MONTH', value_name='VALUE')

df_pivot = df_melted.pivot_table(index=['YEAR', 'REGION', 'MONTH'], columns='PARAMETER', values='VALUE').reset_index()

month_order = {m: i for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}
df_pivot['MONTH_NUM'] = df_pivot['MONTH'].map(month_order)

df_pivot = df_pivot.sort_values(by=['REGION', 'YEAR', 'MONTH_NUM'])

df_pivot['DATE'] = pd.to_datetime(df_pivot['YEAR'].astype(str) + '-' + (df_pivot['MONTH_NUM'] + 1).astype(str) + '-01')

df_pivot['REGION_YEAR_MONTH'] = df_pivot['REGION'] + '-' + df_pivot['YEAR'].astype(str) + '-' + df_pivot['MONTH']

month_encoder = LabelEncoder()
df_pivot['MONTH_ENCODED'] = month_encoder.fit_transform(df_pivot['MONTH'])

region_encoder = LabelEncoder()
df_pivot['REGION_ENCODED'] = region_encoder.fit_transform(df_pivot['REGION'])

features = ['PS', 'RH2M', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'MONTH_ENCODED', 'REGION_ENCODED']
X = df_pivot[features].values
y = df_pivot['PRECTOTCORR'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

time_steps = 6

def create_sequences(X, y, time_steps=6):
    Xs, ys = [], []
    indices = []
    
    df_groups = df_pivot.reset_index().groupby('REGION')
    
    for _, group in df_groups:
        group_indices = group.index.tolist()
        group_X = X_scaled[group_indices]
        group_y = y_scaled[group_indices]
        
        for i in range(len(group_X) - time_steps):
            Xs.append(group_X[i:(i + time_steps)])
            ys.append(group_y[i + time_steps])
            indices.append(group_indices[i + time_steps])
    
    return np.array(Xs), np.array(ys), np.array(indices)

X_seq, y_seq, indices = create_sequences(X_scaled, y_scaled, time_steps)

from sklearn.model_selection import train_test_split

sequence_regions = []
for idx in indices:
    region = df_pivot.iloc[idx]['REGION']
    sequence_regions.append(region)

sequence_regions = np.array(sequence_regions)

X_train, X_test = [], []
y_train, y_test = [], []
indices_train, indices_test = [], []

for region in np.unique(sequence_regions):
    region_mask = (sequence_regions == region)
    
    X_region = X_seq[region_mask]
    y_region = y_seq[region_mask]
    indices_region = indices[region_mask]
    
    if len(X_region) > 5:
        try:
            y_categories = pd.qcut(y_region, q=4, labels=False, duplicates='drop')
            X_train_region, X_test_region, y_train_region, y_test_region, indices_train_region, indices_test_region = train_test_split(
                X_region, y_region, indices_region, test_size=0.2, random_state=42, stratify=y_categories
            )
        except ValueError:
            X_train_region, X_test_region, y_train_region, y_test_region, indices_train_region, indices_test_region = train_test_split(
                X_region, y_region, indices_region, test_size=0.2, random_state=42
            )
    else:
        X_train_region, X_test_region, y_train_region, y_test_region, indices_train_region, indices_test_region = train_test_split(
            X_region, y_region, indices_region, test_size=0.2, random_state=42
        )
    
    X_train.append(X_train_region)
    X_test.append(X_test_region)
    y_train.append(y_train_region)
    y_test.append(y_test_region)
    indices_train.append(indices_train_region)
    indices_test.append(indices_test_region)

X_train = np.concatenate(X_train)
X_test = np.concatenate(X_test)
y_train = np.concatenate(y_train)
y_test = np.concatenate(y_test)
indices_train = np.concatenate(indices_train)
indices_test = np.concatenate(indices_test)

print("Train-test split statistics:")
for region in np.unique(sequence_regions):
    train_count = np.sum(np.array([df_pivot.iloc[idx]['REGION'] == region for idx in indices_train]))
    test_count = np.sum(np.array([df_pivot.iloc[idx]['REGION'] == region for idx in indices_test]))
    print(f"Region {region}: {train_count} training samples, {test_count} validation samples")

model = AdvancedAttentiveRFNetwork(
    n_estimators=300,
    max_depth=50,
    min_samples_split=5,
    input_size=X_train.shape[2],
    time_steps=time_steps,
    hidden_size=128,
    num_heads=4,
    dropout=0.2
)

print("Training Advanced AttentiveRFNetwork...")
model.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.1)

y_pred_train, _ = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_pred_train)
print(f"Training MSE: {train_mse:.4f}")

y_pred_test, attention_weights = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print(f"Test MSE: {test_mse:.4f}")

def generate_predictions(model, df, time_steps=6):
    predictions = []
    actuals = []
    region_year_months = []
    dates = []
    temporal_attention_maps = []
    feature_attention_maps = []
    self_attention_maps = []
    
    for region in df_pivot['REGION'].unique():
        region_data = df_pivot[df_pivot['REGION'] == region].sort_values(['YEAR', 'MONTH_NUM'])
        
        if len(region_data) <= time_steps:
            continue
        
        region_features = region_data[features].values
        region_features_scaled = scaler_X.transform(region_features)
        
        for i in range(time_steps, len(region_data)):
            sequence = region_features_scaled[i-time_steps:i]
            sequence = sequence.reshape(1, time_steps, -1)
            
            pred_scaled, attention_weights = model.predict(sequence)
            pred_scaled = pred_scaled[0]
            
            pred = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            actual = region_data.iloc[i]['PRECTOTCORR']
            
            predictions.append(pred)
            actuals.append(actual)
            region_year_months.append(region_data.iloc[i]['REGION_YEAR_MONTH'])
            dates.append(region_data.iloc[i]['DATE'])
            
            if attention_weights is not None:
                temporal_weights, feature_weights, self_attn_weights = attention_weights
                temporal_attention_maps.append(temporal_weights)
                feature_attention_maps.append(feature_weights)
                self_attention_maps.append(self_attn_weights)
            else:
                temporal_attention_maps.append(np.ones((1, time_steps, 1)) / time_steps)
                feature_attention_maps.append(np.ones((1, time_steps, len(features), 1)) / len(features))
                self_attention_maps.append(np.ones((1, 4, time_steps, time_steps)) / time_steps)
    
    return predictions, actuals, region_year_months, dates, temporal_attention_maps, feature_attention_maps, self_attention_maps

predictions, actuals, region_year_months, dates, temporal_attention_maps, feature_attention_maps, self_attention_maps = generate_predictions(model, df_pivot, time_steps)

results_df = pd.DataFrame({
    'Date': dates,
    'Region_Year_Month': region_year_months,
    'Actual': actuals,
    'Predicted': predictions
})

for i in range(len(results_df)):
    results_df.loc[i, 'Temporal_Attention'] = str(temporal_attention_maps[i].tolist())
    results_df.loc[i, 'Feature_Attention'] = str(feature_attention_maps[i].tolist())
    results_df.loc[i, 'Self_Attention'] = str(self_attention_maps[i].tolist())

results_df = results_df.sort_values('Date')

mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
r2 = r2_score(results_df['Actual'], results_df['Predicted'])

print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R² Score: {r2:.4f}')

fig = plt.figure(figsize=(20, 10))
plt.plot(results_df['Date'], results_df['Actual'], 'o-', markersize=5, color='#1f77b4', alpha=0.8, label='Actual')
plt.plot(results_df['Date'], results_df['Predicted'], 'o-', markersize=5, color='#ff7f0e', alpha=0.8, label='Predicted')

plt.axhline(y=5, color='#2ca02c', linestyle='--', alpha=0.7, label='Light (5 mm)')
plt.axhline(y=10, color='#9467bd', linestyle='--', alpha=0.7, label='Moderate (10 mm)')
plt.axhline(y=15, color='#8c564b', linestyle='--', alpha=0.7, label='Heavy (15 mm)')
plt.axhline(y=20, color='#d62728', linestyle='--', alpha=0.7, label='Extreme (20 mm)')

plt.title('Precipitation Forecast: Actual vs Predicted', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Precipitation (mm)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.xticks(rotation=90, fontsize=8)
plt.tick_params(axis='y', labelsize=12)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_fig(fig, 'adv_precipitation_time_series_forecast.png')

def plot_attention_heatmaps():
    regions = results_df['Region_Year_Month'].str.split('-', expand=True)[0].unique()
    n_regions = len(regions)
    
    fig = plt.figure(figsize=(20, 20))
    
    for i, region in enumerate(regions):
        region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
        
        if len(region_data) == 0:
            continue
        
        sample_size = min(len(region_data), 10)
        sample_indices = np.linspace(0, len(region_data)-1, sample_size, dtype=int)
        
        sampled_data = region_data.iloc[sample_indices]
        
        temporal_attention_data = np.ones((sample_size, time_steps)) / time_steps
        valid_temporal_samples = 0
        
        for j, idx in enumerate(sample_indices):
            try:
                att_str = region_data.iloc[idx]['Temporal_Attention']
                if isinstance(att_str, float) or not isinstance(att_str, str):
                    continue
                
                att_list = json.loads(att_str.replace("'", "\""))
                if isinstance(att_list, list) and len(att_list) > 0:
                    row_data = []
                    for t in range(time_steps):
                        if t < len(att_list[0]) and isinstance(att_list[0][t], list) and len(att_list[0][t]) > 0:
                            row_data.append(att_list[0][t][0])
                        else:
                            row_data.append(1.0/time_steps)
                    
                    temporal_attention_data[j] = row_data
                    valid_temporal_samples += 1
            except Exception as e:
                pass
        
        if valid_temporal_samples > 0:
            plt.subplot(n_regions, 3, i*3+1)
            im = plt.imshow(temporal_attention_data, aspect='auto', cmap='YlOrRd')
            plt.title(f'Region: {region} - Self Attention (Sample {j+1})', fontsize=12)
            plt.xlabel('Sequence Position (Query)', fontsize=10)
            plt.ylabel('Sequence Position (Key)', fontsize=10)
            plt.colorbar(label='Attention Weight')
            plt.xticks(range(time_steps), [f't-{time_steps-i}' for i in range(time_steps)])
            plt.yticks(range(time_steps), [f't-{time_steps-i}' for i in range(time_steps)])
    
    plt.tight_layout()
    save_fig(fig, 'adv_attention_heatmaps.png', bbox_inches='tight')

plot_attention_heatmaps()

regions = results_df['Region_Year_Month'].str.split('-', expand=True)[0].unique()
n_regions = len(regions)

fig, axes = plt.subplots(n_regions, 1, figsize=(28, 5*n_regions), sharex=True)

for i, region in enumerate(regions):
    region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
    
    ax = axes[i] if n_regions > 1 else axes
    
    ax.plot(region_data['Date'], region_data['Actual'], 'o-', markersize=5, color='#1f77b4', label='Actual')
    ax.plot(region_data['Date'], region_data['Predicted'], 'o-', markersize=5, color='#ff7f0e', label='Predicted')
    
    ax.axhline(y=5, color='#2ca02c', linestyle='--', alpha=0.7, label='Light (5 mm)')
    ax.axhline(y=10, color='#9467bd', linestyle='--', alpha=0.7, label='Moderate (10 mm)')
    ax.axhline(y=15, color='#8c564b', linestyle='--', alpha=0.7, label='Heavy (15 mm)')
    ax.axhline(y=20, color='#d62728', linestyle='--', alpha=0.7, label='Extreme (20 mm)')
    
    ax.set_title(f'Region: {region}', fontsize=14)
    ax.set_ylabel('Precipitation (mm)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=10)

plt.xlabel('Date', fontsize=14)
plt.tight_layout(pad=3.0)
plt.subplots_adjust(bottom=0.15)
save_fig(fig, 'adv_precipitation_forecast_by_region.png', bbox_inches='tight')

def plot_feature_importance_by_region():
    regions = results_df['Region_Year_Month'].str.split('-', expand=True)[0].unique()
    feature_importance_data = {}
    
    for region in regions:
        region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
        feature_weights = np.zeros(len(features))
        count = 0
        
        for idx in range(len(region_data)):
            try:
                att_str = region_data.iloc[idx]['Feature_Attention']
                
                if isinstance(att_str, float) or (isinstance(att_str, str) and att_str.strip().startswith("[")):
                    continue
                
                try:
                    att_list = json.loads(att_str.replace("'", "\""))
                    if isinstance(att_list, list) and len(att_list) > 0 and isinstance(att_list[0], list):
                        for k in range(len(features)):
                            feature_sum = 0.0
                            valid_values = 0
                            for t in range(time_steps):
                                if t < len(att_list[0]) and k < len(att_list[0][t]) and isinstance(att_list[0][t][k], list):
                                    if len(att_list[0][t][k]) > 0:
                                        feature_sum += att_list[0][t][k][0]
                                        valid_values += 1
                            if valid_values > 0:
                                feature_weights[k] += feature_sum / valid_values
                    count += 1
                except json.JSONDecodeError:
                    continue
                
            except Exception as e:
                if idx % 20 == 0:
                    print(f"Error processing feature attention for region {region}, index {idx}")
                
        if count > 0:
            feature_weights /= count
            feature_importance_data[region] = feature_weights
    
    if not feature_importance_data:
        for region in regions:
            feature_importance_data[region] = np.ones(len(features)) / len(features)
            
    fig = plt.figure(figsize=(15, 10))
    x = np.arange(len(features))
    width = 0.8 / len(regions)
    
    for i, (region, importance) in enumerate(feature_importance_data.items()):
        plt.bar(x + i*width - 0.4 + width/2, importance, width, label=region)
    
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance Score', fontsize=14)
    plt.title('Feature Importance by Region', fontsize=16)
    plt.xticks(x, features, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'adv_feature_importance_by_region.png')

plot_feature_importance_by_region()

def plot_temporal_importance_by_region():
    regions = results_df['Region_Year_Month'].str.split('-', expand=True)[0].unique()
    temporal_importance_data = {}
    
    for region in regions:
        region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
        temporal_weights = np.zeros(time_steps)
        count = 0
        
        for idx in range(len(region_data)):
            try:
                att_str = region_data.iloc[idx]['Temporal_Attention']
                
                if isinstance(att_str, float) or not isinstance(att_str, str):
                    continue
                
                try:
                    att_list = json.loads(att_str.replace("'", "\""))
                    if isinstance(att_list, list) and len(att_list) > 0:
                        for t in range(min(time_steps, len(att_list[0]))):
                            if isinstance(att_list[0][t], list) and len(att_list[0][t]) > 0:
                                temporal_weights[t] += att_list[0][t][0]
                        count += 1
                except json.JSONDecodeError:
                    continue
                
            except Exception as e:
                if idx % 20 == 0:
                    print(f"Error processing temporal attention for region {region}, index {idx}")
                
        if count > 0:
            temporal_weights /= count
            temporal_importance_data[region] = temporal_weights
    
    if not temporal_importance_data:
        for region in regions:
            temporal_importance_data[region] = np.ones(time_steps) / time_steps
            
    fig = plt.figure(figsize=(12, 8))
    x = np.arange(time_steps)
    
    for region, importance in temporal_importance_data.items():
        plt.plot(x, importance, 'o-', markersize=8, linewidth=2, label=region)
    
    plt.xlabel('Time Steps (Past to Present)', fontsize=14)
    plt.ylabel('Attention Weight', fontsize=14)
    plt.title('Temporal Attention by Region', fontsize=16)
    plt.xticks(x, [f't-{time_steps-i}' for i in range(time_steps)])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'adv_temporal_attention_by_region.png')

plot_temporal_importance_by_region()

def plot_self_attention_patterns():
    fig = plt.figure(figsize=(20, 15))
    
    regions = results_df['Region_Year_Month'].str.split('-', expand=True)[0].unique()
    
    for i, region in enumerate(regions):
        region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
        
        if len(region_data) < 5:
            continue
            
        sample_idx = len(region_data) // 2
        
        try:
            att_str = region_data.iloc[sample_idx]['Self_Attention']
            if isinstance(att_str, float) or not isinstance(att_str, str):
                continue
                
            att_list = json.loads(att_str.replace("'", "\""))
            if not (isinstance(att_list, list) and len(att_list) > 0 and len(att_list[0]) > 0):
                continue
                
            heads = min(4, len(att_list[0]))
            
            for h in range(heads):
                plt.subplot(len(regions), heads, i*heads + h + 1)
                
                if h < len(att_list[0]):
                    attn_matrix = np.array(att_list[0][h])
                    plt.imshow(attn_matrix, cmap='viridis', aspect='auto')
                    plt.colorbar(shrink=0.8)
                    plt.title(f'Region: {region}, Head {h+1}', fontsize=10)
                    plt.xticks(range(time_steps), [f't-{time_steps-j}' for j in range(time_steps)], fontsize=8)
                    plt.yticks(range(time_steps), [f't-{time_steps-j}' for j in range(time_steps)], fontsize=8)
                    plt.xlabel('Key position', fontsize=8)
                    plt.ylabel('Query position', fontsize=8)
        except Exception as e:
            print(f"Error plotting self-attention for region {region}: {e}")
            
    plt.tight_layout()
    save_fig(fig, 'adv_self_attention_patterns.png')
    
plot_self_attention_patterns()

results_df.to_csv('adv_precipitation_forecast_results.csv', index=False)

print(f"Final model performance metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

fig = plt.figure(figsize=(15, 10))
plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.6)
plt.plot([0, max(results_df['Actual'])], [0, max(results_df['Actual'])], 'r--')
plt.xlabel('Actual Precipitation (mm)', fontsize=14)
plt.ylabel('Predicted Precipitation (mm)', fontsize=14)
plt.title('Actual vs Predicted Precipitation Scatter Plot', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_actual_vs_predicted_scatter.png')

high_error_threshold = np.percentile(np.abs(results_df['Actual'] - results_df['Predicted']), 90)
high_error_samples = results_df[np.abs(results_df['Actual'] - results_df['Predicted']) > high_error_threshold]

print(f"Number of samples with high prediction error: {len(high_error_samples)}")
print(f"Error threshold: {high_error_threshold:.4f} mm")

regions_error_count = high_error_samples['Region_Year_Month'].str.split('-', expand=True)[0].value_counts()
print("\nHigh error counts by region:")
print(regions_error_count)

fig = plt.figure(figsize=(10, 6))
regions_error_count.plot(kind='bar')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of High Error Predictions', fontsize=12)
plt.title('High Error Predictions by Region', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_high_error_by_region.png')

months_error_count = high_error_samples['Region_Year_Month'].str.split('-', expand=True)[2].value_counts()
print("\nHigh error counts by month:")
print(months_error_count)

fig = plt.figure(figsize=(10, 6))
sorted_months = sorted(months_error_count.index, key=lambda x: month_order.get(x, 0))
months_error_count_sorted = months_error_count.reindex(sorted_months)
months_error_count_sorted.plot(kind='bar')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of High Error Predictions', fontsize=12)
plt.title('High Error Predictions by Month', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_high_error_by_month.png')

metrics_by_region = []
for region in regions:
    region_data = results_df[results_df['Region_Year_Month'].str.startswith(region)]
    
    region_mse = mean_squared_error(region_data['Actual'], region_data['Predicted'])
    region_rmse = np.sqrt(region_mse)
    region_mae = mean_absolute_error(region_data['Actual'], region_data['Predicted'])
    region_r2 = r2_score(region_data['Actual'], region_data['Predicted'])
    
    metrics_by_region.append({
        'Region': region,
        'MSE': region_mse,
        'RMSE': region_rmse,
        'MAE': region_mae,
        'R²': region_r2
    })

metrics_df = pd.DataFrame(metrics_by_region)
print(metrics_df)

fig = plt.figure(figsize=(12, 8))
metrics_to_plot = ['RMSE', 'MAE']
colors = ['#1f77b4', '#ff7f0e']
bar_width = 0.35
index = np.arange(len(regions))

for i, metric in enumerate(metrics_to_plot):
    plt.bar(index + i*bar_width, metrics_df[metric], bar_width, 
            label=metric, color=colors[i], alpha=0.8)

plt.xlabel('Region', fontsize=14)
plt.ylabel('Error Value (mm)', fontsize=14)
plt.title('Advanced Model Performance by Region', fontsize=16)
plt.xticks(index + bar_width/2, regions, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_model_performance_by_region.png')

feature_importances = model.rf.feature_importances_
feature_names = []

for t in range(time_steps):
    for f in features:
        feature_names.append(f"t-{time_steps-t}_{f}")

sorted_indices = np.argsort(feature_importances)[::-1]
top_features = [feature_names[i] for i in sorted_indices[:20]]
top_importances = [feature_importances[i] for i in sorted_indices[:20]]

fig = plt.figure(figsize=(14, 8))
plt.barh(range(len(top_features)), top_importances, color='forestgreen', alpha=0.7)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importance', fontsize=14)
plt.title('Top 20 Features by Random Forest Importance', fontsize=16)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_top_features_importance.png')

aggregated_importance = {}
for i, name in enumerate(feature_names):
    feature = name.split('_')[1]
    if feature not in aggregated_importance:
        aggregated_importance[feature] = 0
    aggregated_importance[feature] += feature_importances[i]

sorted_agg = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
agg_features = [x[0] for x in sorted_agg]
agg_importances = [x[1] for x in sorted_agg]

fig = plt.figure(figsize=(12, 8))
plt.bar(range(len(agg_features)), agg_importances, color='navy', alpha=0.7)
plt.xticks(range(len(agg_features)), agg_features, rotation=45)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Aggregated Importance', fontsize=14)
plt.title('Features Ranked by Aggregated Random Forest Importance', fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_aggregated_feature_importance.png')

importance_from_attention = {}
for feature_idx, feature_name in enumerate(features):
    importance_from_attention[feature_name] = 0
    count = 0
    
    for i in range(len(results_df)):
        try:
            att_str = results_df.iloc[i]['Feature_Attention']
            if isinstance(att_str, float) or not isinstance(att_str, str):
                continue
                
            att_list = json.loads(att_str.replace("'", "\""))
            if isinstance(att_list, list) and len(att_list) > 0:
                for t in range(time_steps):
                    if (t < len(att_list[0]) and 
                        feature_idx < len(att_list[0][t]) and 
                        isinstance(att_list[0][t][feature_idx], list) and 
                        len(att_list[0][t][feature_idx]) > 0):
                        importance_from_attention[feature_name] += att_list[0][t][feature_idx][0]
                        count += 1
        except:
            continue
            
    if count > 0:
        importance_from_attention[feature_name] /= count

sorted_att_importance = sorted(importance_from_attention.items(), key=lambda x: x[1], reverse=True)
att_features = [x[0] for x in sorted_att_importance]
att_values = [x[1] for x in sorted_att_importance]

fig = plt.figure(figsize=(12, 8))
plt.bar(range(len(att_features)), att_values, color='darkred', alpha=0.7)
plt.xticks(range(len(att_features)), att_features, rotation=45)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Attention Weight', fontsize=14)
plt.title('Features Ranked by Attention Mechanism', fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_attention_feature_importance.png')

fig = plt.figure(figsize=(14, 8))
width = 0.35

common_features = features

rf_values = np.array([aggregated_importance.get(f, 0) for f in common_features])
att_values = np.array([importance_from_attention.get(f, 0) for f in common_features])

rf_sum = rf_values.sum()
if rf_sum > 0:
    normalized_rf = rf_values / rf_sum
else:
    normalized_rf = np.ones_like(rf_values) / len(rf_values)

att_sum = att_values.sum()
if att_sum > 0:
    normalized_att = att_values / att_sum
else:
    normalized_att = np.ones_like(att_values) / len(att_values)

x = np.arange(len(common_features))
plt.bar(x - width/2, normalized_rf, width, label='Random Forest', color='forestgreen', alpha=0.7)
plt.bar(x + width/2, normalized_att, width, label='Attention Mechanism', color='darkred', alpha=0.7)

plt.xlabel('Features', fontsize=14)
plt.ylabel('Normalized Importance', fontsize=14)
plt.title('Feature Importance: Random Forest vs Attention Mechanism', fontsize=16)
plt.xticks(x, features, rotation=45)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_vs_attention_importance.png')

temporal_importance = np.zeros(time_steps)
count = 0

for i in range(len(results_df)):
    try:
        att_str = results_df.iloc[i]['Temporal_Attention']
        if isinstance(att_str, float) or not isinstance(att_str, str):
            continue
            
        att_list = json.loads(att_str.replace("'", "\""))
        if isinstance(att_list, list) and len(att_list) > 0:
            for t in range(time_steps):
                if t < len(att_list[0]) and isinstance(att_list[0][t], list) and len(att_list[0][t]) > 0:
                    temporal_importance[t] += att_list[0][t][0]
            count += 1
    except:
        continue
        
if count > 0:
    temporal_importance /= count

fig = plt.figure(figsize=(10, 6))
plt.bar(range(time_steps), temporal_importance, color='purple', alpha=0.7)
plt.xlabel('Time Steps (Past to Present)', fontsize=14)
plt.ylabel('Average Attention Weight', fontsize=14)
plt.title('Temporal Importance Across All Predictions', fontsize=16)
plt.xticks(range(time_steps), [f't-{time_steps-i}' for i in range(time_steps)])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig(fig, 'adv_temporal_importance.png')

def compare_baseline_vs_advanced():
    try:
        baseline_results = pd.read_csv('rf_precipitation_forecast_results.csv')
        advanced_results = results_df
        
        regions = advanced_results['Region_Year_Month'].str.split('-', expand=True)[0].unique()
        
        comparison_metrics = []
        
        for region in regions:
            baseline_region = baseline_results[baseline_results['Region_Year_Month'].str.startswith(region)]
            advanced_region = advanced_results[advanced_results['Region_Year_Month'].str.startswith(region)]
            
            if len(baseline_region) > 0 and len(advanced_region) > 0:
                common_dates = set(baseline_region['Date']).intersection(set(advanced_region['Date']))
                
                if len(common_dates) > 0:
                    baseline_filtered = baseline_region[baseline_region['Date'].isin(common_dates)]
                    advanced_filtered = advanced_region[advanced_region['Date'].isin(common_dates)]
                    
                    baseline_filtered = baseline_filtered.sort_values('Date')
                    advanced_filtered = advanced_filtered.sort_values('Date')
                    
                    baseline_rmse = np.sqrt(mean_squared_error(baseline_filtered['Actual'], baseline_filtered['Predicted']))
                    advanced_rmse = np.sqrt(mean_squared_error(advanced_filtered['Actual'], advanced_filtered['Predicted']))
                    
                    baseline_mae = mean_absolute_error(baseline_filtered['Actual'], baseline_filtered['Predicted'])
                    advanced_mae = mean_absolute_error(advanced_filtered['Actual'], advanced_filtered['Predicted'])
                    
                    baseline_r2 = r2_score(baseline_filtered['Actual'], baseline_filtered['Predicted'])
                    advanced_r2 = r2_score(advanced_filtered['Actual'], advanced_filtered['Predicted'])
                    
                    improvement_rmse = ((baseline_rmse - advanced_rmse) / baseline_rmse) * 100
                    improvement_mae = ((baseline_mae - advanced_mae) / baseline_mae) * 100
                    improvement_r2 = advanced_r2 - baseline_r2
                    
                    comparison_metrics.append({
                        'Region': region,
                        'Baseline_RMSE': baseline_rmse,
                        'Advanced_RMSE': advanced_rmse,
                        'RMSE_Improvement_%': improvement_rmse,
                        'Baseline_MAE': baseline_mae,
                        'Advanced_MAE': advanced_mae,
                        'MAE_Improvement_%': improvement_mae,
                        'Baseline_R2': baseline_r2,
                        'Advanced_R2': advanced_r2,
                        'R2_Improvement': improvement_r2
                    })
        
        if comparison_metrics:
            comparison_df = pd.DataFrame(comparison_metrics)
            comparison_df.to_csv('model_comparison_results.csv', index=False)
            
            fig = plt.figure(figsize=(15, 10))
            
            index = np.arange(len(comparison_df))
            bar_width = 0.35
            
            plt.bar(index - bar_width/2, comparison_df['Baseline_RMSE'], bar_width, 
                    label='Baseline Model', color='#1f77b4', alpha=0.7)
            plt.bar(index + bar_width/2, comparison_df['Advanced_RMSE'], bar_width,
                    label='Advanced Model', color='#ff7f0e', alpha=0.7)
            
            plt.xlabel('Region', fontsize=14)
            plt.ylabel('RMSE (mm)', fontsize=14)
            plt.title('Model Comparison: RMSE by Region', fontsize=16)
            plt.xticks(index, comparison_df['Region'], rotation=45)
            plt.legend(fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_fig(fig, 'model_comparison_rmse.png')
            
            fig = plt.figure(figsize=(15, 8))
            plt.bar(index, comparison_df['RMSE_Improvement_%'], color='green', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Region', fontsize=14)
            plt.ylabel('RMSE Improvement (%)', fontsize=14)
            plt.title('RMSE Improvement with Advanced Model', fontsize=16)
            plt.xticks(index, comparison_df['Region'], rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_fig(fig, 'model_improvement_percentage.png')
            
            print("\nModel Comparison Results:")
            print(comparison_df)
            
            overall_baseline_rmse = np.sqrt(mean_squared_error(baseline_results['Actual'], baseline_results['Predicted']))
            overall_advanced_rmse = np.sqrt(mean_squared_error(advanced_results['Actual'], advanced_results['Predicted']))
            overall_improvement = ((overall_baseline_rmse - overall_advanced_rmse) / overall_baseline_rmse) * 100
            
            print(f"\nOverall RMSE Improvement: {overall_improvement:.2f}%")
            print(f"Baseline Overall RMSE: {overall_baseline_rmse:.4f}")
            print(f"Advanced Overall RMSE: {overall_advanced_rmse:.4f}")
            
            return comparison_df
    except Exception as e:
        print(f"Couldn't compare models: {e}")
        return None

comparison_results = compare_baseline_vs_advanced()

print("Advanced Self-Attention Model analysis completed!")
