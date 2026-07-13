import requests
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from skyfield.api import EarthSatellite, load, wgs84
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from pytz import timezone

class SatelliteVisualizer:
    # ... 保持不变 ...
    def __init__(self):
        self.satellites = []
        self.ts = load.timescale()
        self.load_satellites()
        self.beijing_tz = timezone('Asia/Shanghai')
        
    def load_satellites(self):
        url = "https://spacemapper.cn/cgi/orbital/aoe/list?format=tle&inclinationMin=86&constellationId=101&lastUpdate=7&altitude=LEO&channel=INTERNATIONAL&rid=a9492a63d8de4591b4b0c1fe1f67c52f"
        try:
            print("正在获取TLE数据...")
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            data = response.text.strip().split('\n')
            data = [line for line in data if line.strip()]
            i = 0
            while i < len(data):
                line = data[i].strip()
                if line.startswith('1 ') and i + 1 < len(data):
                    line1, line2 = line, data[i+1].strip()
                    if line2.startswith('2 '):
                        try:
                            norad_id = line1.split()[1]
                            name = f"Satellite-{norad_id}"
                        except:
                            name = f"SAT-{len(self.satellites)+1}"
                        try:
                            satellite = EarthSatellite(line1, line2, name, self.ts)
                            self.satellites.append(satellite)
                            i += 2
                        except:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1

            # ---------- 新增名称映射 ----------
            sat_name_map = {
                40: "A47", 38: "A48", 39: "A49", 37: "A50", 36: "A51",
                35: "A52", 34: "A53", 33: "A54", 32: "A55", 31: "A56",
                26: "A37", 27: "A38", 28: "A39", 29: "A40", 30: "A41",
                51: "A42", 52: "A43", 53: "A44", 54: "A45", 55: "A46",
                11: "A17", 19: "A18", 20: "A19", 18: "A20", 16: "A21",
                15: "A22", 14: "A23", 17: "A24", 13: "A25", 12: "A26",
                9: "A07", 10: "A08", 2: "A09", 1: "A10", 3: "A11",
                4: "A12", 5: "A13", 8: "A14", 6: "A15", 7: "A16",
                21: "A27", 22: "A28", 23: "A29", 24: "A30", 25: "A31",
                41: "A32", 42: "A33", 43: "A34", 44: "A35", 45: "A36",
                46: "A57", 47: "A58", 48: "A59", 49: "A60", 50: "A61",
                56: "A62", 57: "A63", 58: "A64", 59: "A65", 60: "A66"
            }
            for idx, sat in enumerate(self.satellites):
                seq = idx + 1  # 序号从1开始
                if seq in sat_name_map:
                    sat.name = sat_name_map[seq]   # 直接修改 EarthSatellite 的 name 属性
            # ----------------------------------

            print(f"成功加载 {len(self.satellites)} 颗卫星")
        except Exception as e:
            print(f"获取TLE数据失败: {e}")

    def calculate_beam_coverage(self, sat_alt):
        R_earth, h = 6371.0, sat_alt
        beam_half_angle, min_elevation = np.radians(18), np.radians(10)
        R_sat = R_earth + h
        sin_rho = np.cos(min_elevation) * R_earth / R_sat
        rho = 0 if sin_rho >= 1 else np.arcsin(sin_rho)
        central_angle = np.pi/2 - min_elevation - rho
        sin_beam = (R_sat / R_earth) * np.sin(beam_half_angle)
        if sin_beam > 1:
            beam_central_angle = np.arccos(R_earth / R_sat)
        else:
            alpha = np.arcsin(sin_beam)
            beam_central_angle = np.pi - beam_half_angle - alpha
        coverage_angle = min(central_angle, beam_central_angle)
        return np.degrees(coverage_angle), coverage_angle * R_earth

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        cos_d = np.clip(np.sin(lat1_rad)*np.sin(lat2_rad) + np.cos(lat1_rad)*np.cos(lat2_rad)*np.cos(lon1_rad-lon2_rad), -1, 1)
        return np.degrees(np.arccos(cos_d))

    def generate_coverage_circle(self, center_lat, center_lon, radius_deg, num_points=50):
        lat_center, lon_center = np.radians(center_lat), np.radians(center_lon)
        angles = np.linspace(0, 2*np.pi, num_points)
        lats, lons = [], []
        for angle in angles:
            lat_point = np.arcsin(np.sin(lat_center)*np.cos(np.radians(radius_deg)) + np.cos(lat_center)*np.sin(np.radians(radius_deg))*np.cos(angle))
            lon_point = lon_center + np.arctan2(np.sin(angle)*np.sin(np.radians(radius_deg))*np.cos(lat_center), np.cos(np.radians(radius_deg)) - np.sin(lat_center)*np.sin(lat_point))
            lats.append(np.degrees(lat_point))
            lons.append(np.degrees(lon_point))
        lats.append(lats[0]); lons.append(lons[0])
        return lats, lons

    def get_satellite_positions(self, target_time=None):
        if not self.satellites:
            return {'lats':[],'lons':[],'names':[],'alts':[],'coverages':[],'coverage_circles':[],'ground_radii':[]}
        if target_time is None:
            now = self.ts.now()
        else:
            now = target_time
        positions = {'lats':[],'lons':[],'names':[],'alts':[],'coverages':[],'coverage_circles':[],'ground_radii':[]}
        for sat in self.satellites:
            try:
                geocentric = sat.at(now)
                subpoint = wgs84.subpoint(geocentric)
                lat, lon, alt = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km
                coverage_radius_deg, ground_radius_km = self.calculate_beam_coverage(alt)
                circle_lats, circle_lons = self.generate_coverage_circle(lat, lon, coverage_radius_deg)
                positions['lats'].append(lat); positions['lons'].append(lon); positions['names'].append(sat.name)
                positions['alts'].append(alt); positions['coverages'].append(coverage_radius_deg)
                positions['ground_radii'].append(ground_radius_km)
                positions['coverage_circles'].append({'lats':circle_lats, 'lons':circle_lons})
            except:
                continue
        return positions

    def query_covering_satellites(self, target_lat, target_lon, target_time=None):
        positions = self.get_satellite_positions(target_time)
        covering = []
        for i in range(len(positions['lats'])):
            distance = self.calculate_distance(target_lat, target_lon, positions['lats'][i], positions['lons'][i])
            if distance <= positions['coverages'][i]:
                covering.append({'index':i+1,'name':positions['names'][i],'distance':distance,
                                 'ground_radius':positions['ground_radii'][i],'sat_lat':positions['lats'][i],
                                 'sat_lon':positions['lons'][i],'alt':positions['alts'][i]})
        covering.sort(key=lambda x: x['distance'])
        return covering

    def create_fullscreen_map(self, query_point=None, covering_indices=None, target_time=None):
        if target_time is None:
            target_time = self.ts.now()
        positions = self.get_satellite_positions(target_time)
        beijing_time = target_time.astimezone(self.beijing_tz).strftime("%Y-%m-%d %H:%M:%S")
        fig = go.Figure()
        if positions['lats']:
            for i in range(len(positions['lats'])):
                lat,lon,alt = positions['lats'][i],positions['lons'][i],positions['alts'][i]
                name,ground_radius = positions['names'][i],positions['ground_radii'][i]
                circle_data = positions['coverage_circles'][i]
                is_covering = covering_indices and (i+1) in covering_indices
                marker_size = 10 if is_covering else 6
                marker_color = '#00FF00' if is_covering else '#FF3333'
                marker_border = '#00CC00' if is_covering else '#CC0000'
                fig.add_trace(go.Scattergeo(
                    lon=[lon], lat=[lat], mode='markers+text',
                    marker=dict(size=marker_size, color=marker_color, symbol='circle',
                               line=dict(width=2 if is_covering else 1, color=marker_border),
                               opacity=1.0 if is_covering else 0.8),
                    text=[positions['names'][i]], textposition="top center",
                    textfont=dict(size=8 if is_covering else 6, color='#00FF00' if is_covering else '#FF6666'),
                    hovertext=f"<b>卫星 #{i+1}</b><br>名称: {name}<br>纬度: {lat:.2f}°<br>经度: {lon:.2f}°<br>高度: {alt:.1f} km<br>覆盖半径: {ground_radius:.0f} km",
                    hoverinfo='text', showlegend=False
                ))
                fig.add_trace(go.Scattergeo(
                    lon=circle_data['lons'], lat=circle_data['lats'], mode='lines',
                    line=dict(width=2 if is_covering else 1,
                             color='rgba(0,255,0,0.5)' if is_covering else 'rgba(100,180,255,0.2)'),
                    fill='toself', fillcolor='rgba(0,255,0,0.1)' if is_covering else 'rgba(100,180,255,0.03)',
                    hoverinfo='skip', showlegend=False, visible=is_covering
                ))
        if query_point:
            fig.add_trace(go.Scattergeo(
                lon=[query_point['lon']], lat=[query_point['lat']], mode='markers',
                marker=dict(size=12, color='#FFFF00', symbol='star', line=dict(width=2, color='#FFA500')),
                name='查询点', hovertext=f"<b>查询点</b><br>纬度: {query_point['lat']:.4f}°<br>经度: {query_point['lon']:.4f}°",
                hoverinfo='text'
            ))
        fig.add_trace(go.Scattergeo(lon=[None],lat=[None],mode='markers',
            marker=dict(size=8,color='#FF3333',symbol='circle',line=dict(width=1,color='#CC0000')),
            name='卫星',showlegend=True))
        if covering_indices:
            fig.add_trace(go.Scattergeo(lon=[None],lat=[None],mode='markers',
                marker=dict(size=8,color='#00FF00',symbol='circle',line=dict(width=1,color='#00CC00')),
                name='覆盖卫星',showlegend=True))
        if query_point:
            fig.add_trace(go.Scattergeo(lon=[None],lat=[None],mode='markers',
                marker=dict(size=10,color='#FFFF00',symbol='star',line=dict(width=1.5,color='#FFA500')),
                name='查询点',showlegend=True))
        title_text = f'🕐 查询时间: {beijing_time} 北京时间'
        fig.update_layout(
            title=dict(text=title_text, x=0.5, y=0.98, font=dict(size=20, color='#CCCCCC'), xanchor='center', yanchor='top'),
            geo=dict(projection_type='equirectangular', showland=True, landcolor='rgb(50,50,50)',
                     coastlinecolor='rgb(120,120,120)', showocean=True, oceancolor='rgb(20,30,45)',
                     showcountries=True, countrycolor='rgb(90,90,90)', showlakes=True, lakecolor='rgb(20,30,45)',
                     showframe=False, bgcolor='rgb(15,20,30)', coastlinewidth=0.8, countrywidth=0.5),
            paper_bgcolor='rgb(15,20,30)', plot_bgcolor='rgb(15,20,30)',
            margin=dict(l=0,r=0,t=40,b=0), showlegend=True,
            legend=dict(x=0.01,y=0.99,xanchor='left',yanchor='top',bgcolor='rgba(20,25,35,0.8)',
                       bordercolor='rgba(100,100,100,0.3)',borderwidth=1,font=dict(color='#CCCCCC',size=10)),
            autosize=True, uirevision='constant', hovermode='closest',
            hoverlabel=dict(bgcolor='rgba(15,20,30,0.95)',font=dict(color='white',size=12),bordercolor='#4FC3F7')
        )
        return fig


app = dash.Dash(__name__)
print("正在初始化卫星可视化系统...")
viz = SatelliteVisualizer()

# 样式常量
PANEL_BG = 'rgba(18,22,32,0.96)'
BORDER_COLOR = 'rgba(255,255,255,0.12)'
ACCENT_BLUE = '#4FC3F7'
ACCENT_GREEN = '#00E676'
ACCENT_RED = '#FF5252'
TEXT_COLOR = '#E0E0E0'
TEXT_SECONDARY = '#B0B0B0'
INPUT_BG = 'rgba(30,35,50,0.8)'

now = datetime.now(timezone('Asia/Shanghai'))
default_date = now.strftime("%Y-%m-%d")
default_time = now.strftime("%H:%M")

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0),
    dcc.Store(id='query-point', data=None),
    dcc.Store(id='covering-indices', data=None),
    dcc.Store(id='show-query-panel', data=False),
    dcc.Store(id='target-timestamp', data=None),
    
    dcc.Graph(id='live-satellite-map', figure=viz.create_fullscreen_map(),
              config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True, 'responsive': True},
              style={'height': '100vh', 'width': '100vw', 'position': 'fixed', 'top': 0, 'left': 0}),
    
    html.Button('🔍 覆盖查询', id='toggle-query-btn',
                style={'position': 'fixed', 'top': '15px', 'right': '15px', 'zIndex': '1000',
                       'padding': '10px 18px', 'backgroundColor': PANEL_BG, 'color': TEXT_COLOR,
                       'border': f'1px solid {BORDER_COLOR}', 'borderRadius': '10px', 'cursor': 'pointer',
                       'fontSize': '14px', 'fontWeight': '500', 'fontFamily': 'Arial, sans-serif',
                       'backdropFilter': 'blur(10px)', 'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
                       'transition': 'all 0.2s ease', 'letterSpacing': '0.5px'}),
    
    html.Div([
        html.Div([
            html.Span('📍 卫星覆盖查询', style={'color': TEXT_COLOR, 'fontSize': '17px', 'fontWeight': '600', 'fontFamily': 'Arial, sans-serif'}),
            html.Button('✕', id='close-panel-btn',
                       style={'background': 'none', 'border': 'none', 'color': TEXT_SECONDARY, 'fontSize': '18px',
                              'cursor': 'pointer', 'padding': '2px 8px', 'borderRadius': '6px',
                              'transition': 'all 0.2s ease'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),
        
        # 时间选择区域（美化重点）
        html.Div([
            html.Div([
                html.Span('🕐', style={'fontSize': '16px', 'marginRight': '8px'}),
                html.Span('查询时间', style={'color': TEXT_COLOR, 'fontSize': '13px', 'fontWeight': '500',
                                             'fontFamily': 'Arial, sans-serif'})
            ], style={'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center'}),
            
            # 日期选择器和时间输入框放在同一行
            html.Div([
                dcc.DatePickerSingle(
                    id='query-date',
                    date=default_date,
                    display_format='YYYY-MM-DD',
                    className='date-picker',
                    style={'flex': '1', 'marginRight': '8px'}
                ),
                html.Div([
                    dcc.Input(
                        id='query-time',
                        type='text',
                        value=default_time,
                        placeholder='HH:MM',
                        className='time-input'
                    )
                ], style={'width': '80px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
            
            # 快捷按钮组
            html.Div([
                html.Button('🔄 现在', id='now-btn', className='time-btn time-btn-primary'),
                html.Button('⏪ -1h', id='minus-1h-btn', className='time-btn'),
                html.Button('⏩ +1h', id='plus-1h-btn', className='time-btn'),
            ], style={'display': 'flex', 'gap': '6px'}),
            
            html.Div(style={'height': '16px'}),  # 间隔
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)', 'margin': '0 0 18px 0'}),
        ], style={
            'backgroundColor': 'rgba(25,30,42,0.5)',
            'borderRadius': '10px',
            'padding': '14px',
            'marginBottom': '8px',
            'border': '1px solid rgba(255,255,255,0.06)'
        }),
        
        # 经纬度输入
        html.Div([
            html.Div([
                html.Label('🌐 纬度', style={'color': TEXT_SECONDARY, 'fontSize': '11px', 'fontWeight': '500', 'marginBottom': '5px',
                                             'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                dcc.Input(id='input-lat', type='number', placeholder='39.9', min=-90, max=90, step=0.1,
                          className='coord-input')
            ], style={'marginBottom': '12px'}),
            html.Div([
                html.Label('🌐 经度', style={'color': TEXT_SECONDARY, 'fontSize': '11px', 'fontWeight': '500', 'marginBottom': '5px',
                                             'textTransform': 'uppercase', 'letterSpacing': '1px'}),
                dcc.Input(id='input-lon', type='number', placeholder='116.4', min=-180, max=180, step=0.1,
                          className='coord-input')
            ], style={'marginBottom': '18px'}),
            
            html.Div([
                html.Button('🔍 查询', id='query-btn', className='action-btn action-btn-primary'),
                html.Button('🗑️ 清除', id='clear-btn', className='action-btn action-btn-danger'),
            ], style={'display': 'flex', 'gap': '8px'}),
        ]),
        
        html.Div(id='query-result', className='result-box')
    ], id='query-panel', style={'position': 'fixed', 'top': '60px', 'right': '15px', 'width': '340px', 'padding': '22px',
               'backgroundColor': PANEL_BG, 'borderRadius': '14px', 'border': f'1px solid {BORDER_COLOR}',
               'zIndex': '999', 'display': 'none', 'backdropFilter': 'blur(20px)',
               'boxShadow': '0 8px 30px rgba(0,0,0,0.4)', 'fontFamily': 'Arial, sans-serif'})
], style={'margin': 0, 'padding': 0, 'height': '100vh', 'width': '100vw', 'overflow': 'hidden', 'backgroundColor': 'rgb(15,20,30)'})


# 回调函数保持不变（略作适配）
@app.callback(
    [Output('query-panel', 'style'), Output('show-query-panel', 'data')],
    Input('toggle-query-btn', 'n_clicks'),
    State('show-query-panel', 'data')
)
def toggle_query_panel(n_clicks, current_state):
    base = {'position': 'fixed', 'top': '60px', 'right': '15px', 'width': '340px', 'padding': '22px',
            'backgroundColor': PANEL_BG, 'borderRadius': '14px', 'border': f'1px solid {BORDER_COLOR}',
            'zIndex': '999', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 8px 30px rgba(0,0,0,0.4)',
            'fontFamily': 'Arial, sans-serif'}
    if n_clicks and n_clicks % 2 == 1:
        return {**base, 'display': 'block'}, True
    return {**base, 'display': 'none'}, False

@app.callback(
    [Output('query-panel', 'style', allow_duplicate=True), Output('show-query-panel', 'data', allow_duplicate=True)],
    Input('close-panel-btn', 'n_clicks'), prevent_initial_call=True
)
def close_query_panel(n_clicks):
    base = {'position': 'fixed', 'top': '60px', 'right': '15px', 'width': '340px', 'padding': '22px',
            'backgroundColor': PANEL_BG, 'borderRadius': '14px', 'border': f'1px solid {BORDER_COLOR}',
            'zIndex': '999', 'backdropFilter': 'blur(20px)', 'boxShadow': '0 8px 30px rgba(0,0,0,0.4)',
            'display': 'none', 'fontFamily': 'Arial, sans-serif'}
    return base, False

@app.callback(
    [Output('query-date', 'date'), Output('query-time', 'value')],
    Input('now-btn', 'n_clicks'), prevent_initial_call=True
)
def set_now(n_clicks):
    now = datetime.now(timezone('Asia/Shanghai'))
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M")

@app.callback(
    [Output('query-date', 'date', allow_duplicate=True), Output('query-time', 'value', allow_duplicate=True)],
    Input('minus-1h-btn', 'n_clicks'),
    [State('query-date', 'date'), State('query-time', 'value')],
    prevent_initial_call=True
)
def minus_one_hour(n_clicks, current_date, current_time):
    if current_date and current_time:
        try:
            dt = datetime.strptime(f"{current_date} {current_time}", "%Y-%m-%d %H:%M")
            dt = timezone('Asia/Shanghai').localize(dt) - timedelta(hours=1)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
        except:
            pass
    return current_date, current_time

@app.callback(
    [Output('query-date', 'date', allow_duplicate=True), Output('query-time', 'value', allow_duplicate=True)],
    Input('plus-1h-btn', 'n_clicks'),
    [State('query-date', 'date'), State('query-time', 'value')],
    prevent_initial_call=True
)
def plus_one_hour(n_clicks, current_date, current_time):
    if current_date and current_time:
        try:
            dt = datetime.strptime(f"{current_date} {current_time}", "%Y-%m-%d %H:%M")
            dt = timezone('Asia/Shanghai').localize(dt) + timedelta(hours=1)
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
        except:
            pass
    return current_date, current_time

@app.callback(
    [Output('query-point', 'data'), Output('covering-indices', 'data'), 
     Output('query-result', 'children'), Output('target-timestamp', 'data')],
    Input('query-btn', 'n_clicks'),
    [State('input-lat', 'value'), State('input-lon', 'value'),
     State('query-date', 'date'), State('query-time', 'value')]
)
def query_satellites(n_clicks, lat, lon, query_date, query_time):
    if not n_clicks or lat is None or lon is None:
        return None, None, html.Div('输入经纬度查询', style={'color': TEXT_SECONDARY, 'textAlign': 'center', 'padding': '10px', 'fontSize': '12px'}), None
    target_time = None
    if query_date and query_time:
        try:
            dt = datetime.strptime(f"{query_date} {query_time}", "%Y-%m-%d %H:%M")
            dt = timezone('Asia/Shanghai').localize(dt)
            target_time = viz.ts.from_datetime(dt)
        except:
            target_time = viz.ts.now()
    else:
        target_time = viz.ts.now()
    covering = viz.query_covering_satellites(lat, lon, target_time)
    query_point = {'lat': lat, 'lon': lon}
    covering_indices = [sat['index'] for sat in covering]
    ts = target_time.astimezone(timezone('Asia/Shanghai')).timestamp()
    if covering:
        result = [
            html.Div([html.Span(f'📍 ({lat:.1f}°, {lon:.1f}°)', style={'color': '#FFD600', 'fontSize': '13px', 'fontWeight': '600'})]),
            html.Div([html.Span(f'✅ 覆盖卫星: {len(covering)} 颗', style={'color': ACCENT_GREEN, 'fontSize': '12px', 'fontWeight': '500'})], style={'marginTop': '8px'}),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)', 'margin': '10px 0'})
        ]
        for sat in covering[:10]:
            result.append(html.Div([
                html.Div([html.Span(f"#{sat['index']}", style={'color': ACCENT_GREEN, 'fontWeight': '700', 'fontSize': '12px'}),
                         html.Span(f" 距离: {sat['distance']:.1f}°", style={'color': TEXT_SECONDARY, 'fontSize': '11px'})], style={'marginBottom': '4px'}),
                html.Div([html.Span(f"🛰️ 高度: {sat['alt']:.0f}km", style={'color': TEXT_SECONDARY, 'fontSize': '10px', 'marginRight': '10px'}),
                         html.Span(f"📡 半径: {sat['ground_radius']:.0f}km", style={'color': TEXT_SECONDARY, 'fontSize': '10px'})]),
                html.Hr(style={'borderColor': 'rgba(255,255,255,0.04)', 'margin': '6px 0'})
            ], style={'padding': '4px 0'}))
        if len(covering) > 10:
            result.append(html.P(f'... 共 {len(covering)} 颗卫星', style={'color': TEXT_SECONDARY, 'fontSize': '11px', 'textAlign': 'center', 'marginTop': '5px'}))
        return query_point, covering_indices, html.Div(result), ts
    return query_point, [], html.Div([
        html.Div([html.Span(f'📍 ({lat:.1f}°, {lon:.1f}°)', style={'color': '#FFD600', 'fontSize': '13px', 'fontWeight': '600'})]),
        html.Div([html.Span('❌ 无卫星覆盖', style={'color': ACCENT_RED, 'fontSize': '13px', 'fontWeight': '500'})], style={'marginTop': '12px', 'textAlign': 'center'})
    ]), ts

@app.callback(
    [Output('query-point', 'data', allow_duplicate=True), Output('covering-indices', 'data', allow_duplicate=True),
     Output('query-result', 'children', allow_duplicate=True), Output('input-lat', 'value'), 
     Output('input-lon', 'value'), Output('target-timestamp', 'data', allow_duplicate=True)],
    Input('clear-btn', 'n_clicks'), prevent_initial_call=True
)
def clear_query(n_clicks):
    return None, None, html.Div('输入经纬度查询', style={'color': TEXT_SECONDARY, 'textAlign': 'center', 'padding': '10px', 'fontSize': '12px'}), None, None, None

@app.callback(
    Output('live-satellite-map', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('query-point', 'data'),
     Input('covering-indices', 'data'),
     Input('target-timestamp', 'data')]
)
def update_map(n_intervals, query_point, covering_indices, target_timestamp):
    target_time = None
    if target_timestamp is not None:
        dt = datetime.fromtimestamp(target_timestamp, tz=timezone('Asia/Shanghai'))
        target_time = viz.ts.from_datetime(dt)
    if query_point is None:
        target_time = None
    return viz.create_fullscreen_map(query_point, covering_indices, target_time)


# 增强的CSS样式
app.index_string = '''<!DOCTYPE html><html><head>{%metas%}<title>🛰️ 卫星覆盖查询系统</title>{%css%}
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{margin:0;padding:0;overflow:hidden;background:rgb(15,20,30);height:100vh;width:100vw;font-family:'Segoe UI',Arial,sans-serif}

/* 按钮悬停效果 */
#toggle-query-btn:hover{background:rgba(30,36,50,0.95)!important;border-color:rgba(255,255,255,0.25)!important;transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,0,0,0.4)!important}
#close-panel-btn:hover{color:#E0E0E0!important;background:rgba(255,255,255,0.08)!important}
#query-btn:hover{background:rgba(79,195,247,0.25)!important;border-color:rgba(79,195,247,0.5)!important}
#clear-btn:hover{background:rgba(255,82,82,0.2)!important;border-color:rgba(255,82,82,0.4)!important}

/* 通用输入框样式 */
.coord-input{
    width:100%; padding:10px 12px; background-color:rgba(30,35,50,0.8); color:#E0E0E0;
    border:1px solid rgba(255,255,255,0.12); border-radius:8px; font-size:13px;
    font-family:Arial,sans-serif; outline:none; transition:all 0.2s ease;
}
.coord-input:focus{border-color:rgba(79,195,247,0.5)!important;box-shadow:0 0 0 2px rgba(79,195,247,0.1)!important}

/* 时间输入框 */
.time-input{
    width:100%; padding:8px 10px; background-color:rgba(30,35,50,0.9); color:#E0E0E0;
    border:1px solid rgba(255,255,255,0.15); border-radius:6px; font-size:13px;
    font-family:'Consolas',monospace; text-align:center; letter-spacing:1px;
    transition:all 0.2s ease;
}
.time-input:focus{border-color:rgba(79,195,247,0.5)!important;box-shadow:0 0 0 2px rgba(79,195,247,0.1)!important}

/* 日期选择器美化 */
.date-picker .DateInput_input{
    background-color:rgba(30,35,50,0.9)!important; color:#E0E0E0!important;
    border:1px solid rgba(255,255,255,0.15)!important; border-radius:6px!important;
    padding:8px 10px!important; font-size:13px!important; font-family:Arial,sans-serif!important;
    line-height:normal!important;
}
.DateInput{width:100%!important}
.SingleDatePickerInput{border:none!important;background:transparent!important}

/* 时间快捷按钮 */
.time-btn{
    flex:1; padding:8px 4px; border-radius:6px; cursor:pointer; font-size:12px;
    font-weight:500; font-family:Arial,sans-serif; transition:all 0.2s ease;
    background:rgba(255,255,255,0.03); color:#B0B0B0;
    border:1px solid rgba(255,255,255,0.1);
}
.time-btn:hover{background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.2); color:#E0E0E0}
.time-btn-primary{background:rgba(79,195,247,0.15); color:#4FC3F7; border-color:rgba(79,195,247,0.3)}
.time-btn-primary:hover{background:rgba(79,195,247,0.25)!important; border-color:rgba(79,195,247,0.5)!important}

/* 操作按钮 */
.action-btn{
    flex:1; padding:10px; border-radius:8px; cursor:pointer; font-size:13px;
    font-weight:500; font-family:Arial,sans-serif; transition:all 0.2s ease;
}
.action-btn-primary{background:rgba(79,195,247,0.15); color:#4FC3F7; border:1px solid rgba(79,195,247,0.3)}
.action-btn-primary:hover{background:rgba(79,195,247,0.25)!important; border-color:rgba(79,195,247,0.5)!important}
.action-btn-danger{background:rgba(255,82,82,0.1); color:#FF5252; border:1px solid rgba(255,82,82,0.2)}
.action-btn-danger:hover{background:rgba(255,82,82,0.2)!important; border-color:rgba(255,82,82,0.4)!important}

/* 结果展示区 */
.result-box{
    color:#E0E0E0; max-height:180px; overflow-y:auto; margin-top:15px;
    padding:12px; background-color:rgba(25,30,42,0.6); border-radius:8px;
    border:1px solid rgba(255,255,255,0.06); font-size:12px; font-family:Arial,sans-serif;
}

/* 滚动条 */
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.2)}
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

# 在 app = dash.Dash(__name__) 附近或之后添加
server = app.server

# 在文件末尾
if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║  🛰️  卫星覆盖查询系统 v3.1        ║")
    print(f"║  📡 已加载 {len(viz.satellites)} 颗卫星           ║")
    print("║  🕐 支持历史时间查询               ║")
    print("║  🔗 http://127.0.0.1:8050         ║")
    print("╚══════════════════════════════════════╝")
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)