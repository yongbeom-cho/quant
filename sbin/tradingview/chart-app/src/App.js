import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ChartComponent from './components/ChartComponent';

const INTERVAL_OPTIONS = [
    { label: '1분', value: 'minute1' }, { label: '3분', value: 'minute3' },
    { label: '5분', value: 'minute5' }, { label: '15분', value: 'minute15' },
    { label: '30분', value: 'minute30' }, { label: '1시간', value: 'minute60' },
    { label: '4시간', value: 'minute240' }, { label: '일봉', value: 'day' },
    { label: '주봉', value: 'week' }, { label: '월봉', value: 'month' },
];

function App() {
    const [data, setData] = useState([]);
    const [tickers, setTickers] = useState([]);
    const [ticker, setTicker] = useState("KRW-BTC");
    const [interval, setInterval] = useState("minute1");
    const [searchTerm, setSearchTerm] = useState("");
    const [isFetching, setIsFetching] = useState(false);
    const [hasMore, setHasMore] = useState(true);

    // 초기 상태: MA 하나 추가됨
    const [indicators, setIndicators] = useState([
        { id: 1, category: 'ma', type: 'sma', period: '20', color: '#f2ff00', width: 2, active: true }
    ]);
    const [debouncedPeriods, setDebouncedPeriods] = useState({});

    const offsetRef = useRef(0);
    const isFetchingRef = useRef(false);

    // 디바운싱: 입력창에 타이핑 할 때마다 재요청 방지
    useEffect(() => {
        const handler = setTimeout(() => {
            const periods = {};
            indicators.forEach(ind => { periods[ind.id] = ind.period; });
            setDebouncedPeriods(periods);
        }, 400);
        return () => clearTimeout(handler);
    }, [indicators]); // indicators 전체 감시

    // 티커 목록 로딩
    useEffect(() => {
        fetch(`http://localhost:8000/api/tickers`)
            .then(res => res.json())
            .then(setTickers)
            .catch(err => console.error(err));
    }, []);

    // 서버로 보낼 설정값 생성
    const activeConfigs = useMemo(() => {
        return indicators.filter(i => i.active).map(i => ({
            ...i,
            period: debouncedPeriods[i.id] || i.period
        }));
    }, [indicators, debouncedPeriods]);

    // 데이터 패칭
    const fetchData = useCallback(async (isInitial = false) => {
        if (isFetchingRef.current || (!isInitial && !hasMore)) return;
        isFetchingRef.current = true;
        setIsFetching(true);
        
        if (isInitial) {
            offsetRef.current = 0;
            setHasMore(true);
        }
        
        try {
            const params = new URLSearchParams();
            // config 포맷: type_period (예: sma_20, ichimoku_9,26,52)
            activeConfigs.forEach(i => params.append("configs", `${i.type}_${i.period}`));
            params.append("ticker", ticker);
            params.append("interval", interval);
            params.append("limit", 500);
            params.append("offset", offsetRef.current);

            const res = await fetch(`http://localhost:8000/api/ohlcv?${params.toString()}`);
            const newData = await res.json();

            if (!newData || newData.length === 0) {
                setHasMore(false);
            } else {
                setData(prev => {
                    const combined = isInitial ? newData : [...newData, ...prev];
                    // 중복 제거 및 정렬
                    return combined
                        .filter((v, i, a) => a.findIndex(t => t.time === v.time) === i)
                        .sort((a, b) => a.time - b.time);
                });
                offsetRef.current += 500;
                if (newData.length < 500) setHasMore(false);
            }
        } catch (error) { 
            console.error(error); 
        } finally { 
            setIsFetching(false); 
            isFetchingRef.current = false; 
        }
    }, [ticker, interval, activeConfigs, hasMore]);

    // 지표 변경시 데이터 리로드
    useEffect(() => {
        fetchData(true);
    }, [ticker, interval, activeConfigs]);

    // 지표 추가 함수
    const addIndicator = (category) => {
        const id = Date.now();
        const base = { id, category, active: true, width: 2 };
        
        // 기본 프리셋
        const presets = {
            'ma': { type: 'sma', period: '20', color: '#f2ff00' },
            'bb': { type: 'bollinger', period: '20', color: '#2196F3' },
            'ichi': { type: 'ichimoku', period: '9,26,52', color: '#9C27B0' },
            'rsi': { type: 'rsi', period: '14', color: '#E91E63' },
            'cci': { type: 'cci', period: '20', color: '#FF9800' },
            'mfi': { type: 'mfi', period: '14', color: '#4CAF50' },
            'atr': { type: 'atr', period: '14', color: '#00BCD4' },
            'dc': { type: 'donchian', period: '20', color: '#795548' },
            'psar': { type: 'psar', period: '0', color: '#FFFFFF', width: 1 },
            'adx': { type: 'adx', period: '14', color: '#FFEB3B' },
            'volma': { type: 'volma', period: '20', color: '#607D8B' }
        };
        setIndicators([...indicators, { ...base, ...presets[category] }]);
    };

    // 지표 속성 업데이트
    const updateIndicator = (id, field, value) => {
        setIndicators(indicators.map(ind => ind.id === id ? { ...ind, [field]: value } : ind));
    };

    return (
        <div style={containerStyle}>
            {/* 왼쪽 사이드바: 지표 제어 */}
            <div style={leftSidebarStyle}>
                <h3 style={sidebarTitle}>지표 설정</h3>
                <div style={addBtnGroup}>
                    {['ma', 'bb', 'ichi', 'rsi', 'cci', 'mfi', 'atr', 'dc', 'psar', 'adx', 'volma'].map(cat => (
                        <button key={cat} onClick={() => addIndicator(cat)} style={addBtnStyle}>{cat.toUpperCase()}</button>
                    ))}
                </div>
                <div style={{ overflowY: 'auto', flex: 1 }}>
                    {indicators.map(ind => (
                        <div key={ind.id} style={{...cardStyle, borderLeft: `4px solid ${ind.color}`, opacity: ind.active ? 1 : 0.6}}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                <div style={{display:'flex', alignItems:'center', gap: '6px'}}>
                                    <input type="checkbox" checked={ind.active} onChange={e => updateIndicator(ind.id, 'active', e.target.checked)} />
                                    
                                    {/* MA인 경우 Dropdown 부활 */}
                                    {ind.category === 'ma' ? (
                                        <select value={ind.type} onChange={e => updateIndicator(ind.id, 'type', e.target.value)} style={miniSelect}>
                                            <option value="sma">SMA</option>
                                            <option value="ema">EMA</option>
                                            <option value="wma">WMA</option>
                                        </select>
                                    ) : (
                                        <span style={indicatorLabel}>{ind.type.toUpperCase()}</span>
                                    )}
                                </div>
                                <button onClick={() => setIndicators(indicators.filter(i => i.id !== ind.id))} style={delBtn}>✕</button>
                            </div>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr 1fr', gap: '5px' }}>
                                {/* PSAR은 파라미터가 없거나 고정이라도 period 인풋은 놔둠 (안쓰더라도 에러방지) */}
                                <input type="text" value={ind.period} onChange={e => updateIndicator(ind.id, 'period', e.target.value)} style={miniInput} />
                                <select value={ind.width} onChange={e => updateIndicator(ind.id, 'width', parseInt(e.target.value))} style={miniSelect}>
                                    {[1,2,3,4,5].map(w => <option key={w} value={w}>{w}px</option>)}
                                </select>
                                <input type="color" value={ind.color} onChange={e => updateIndicator(ind.id, 'color', e.target.value)} style={colorPicker} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 메인 차트 영역 */}
            <div style={chartAreaStyle}>
                <div style={chartHeader}>
                    <div style={{ fontSize: '18px', fontWeight: 'bold' }}>{ticker}</div>
                    <div style={{ display: 'flex', gap: '4px' }}>
                        {INTERVAL_OPTIONS.map(opt => (
                            <button key={opt.value} onClick={() => setInterval(opt.value)} style={{...intervalBtn, backgroundColor: interval === opt.value ? '#2962ff' : '#1e222d', color: interval === opt.value ? '#fff' : '#848e9c'}}>{opt.label}</button>
                        ))}
                    </div>
                </div>
                <div style={{ flex: 1, position: 'relative' }}>
                    <ChartComponent data={data} configs={activeConfigs} onLoadMore={() => fetchData(false)} />
                </div>
            </div>

            {/* 오른쪽 사이드바: 종목 검색 */}
            <div style={rightSidebarStyle}>
                <div style={searchWrapper}>
                    <input type="text" placeholder="심볼 검색" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} style={searchInput} />
                </div>
                <div style={{ flex: 1, overflowY: 'auto' }}>
                    <table style={tickerTable}>
                        <tbody>
                            {tickers.filter(t => t.replace("KRW-","").toLowerCase().includes(searchTerm.toLowerCase())).map(t => (
                                <tr key={t} onClick={() => setTicker(t)} style={{ ...tickerRow, backgroundColor: ticker === t ? '#2b2b43' : 'transparent' }}>
                                    <td style={{ padding: '10px' }}>
                                        <div style={{ fontWeight: 'bold' }}>{t.replace("KRW-", "")}</div>
                                        <div style={{ fontSize: '11px', color: '#848e9c' }}>KRW</div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

// 스타일들
const containerStyle = { display: 'flex', backgroundColor: '#090a0d', height: '100vh', width: '100vw', color: '#d1d4dc', overflow: 'hidden' };
const leftSidebarStyle = { width: '240px', backgroundColor: '#131722', borderRight: '1px solid #2b2b43', padding: '15px', display: 'flex', flexDirection: 'column' };
const chartAreaStyle = { flex: 1, display: 'flex', flexDirection: 'column' };
const rightSidebarStyle = { width: '240px', backgroundColor: '#131722', borderLeft: '1px solid #2b2b43', display: 'flex', flexDirection: 'column' };
const sidebarTitle = { fontSize: '14px', marginBottom: '15px', color: '#848e9c' };
const addBtnGroup = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', marginBottom: '15px' };
const addBtnStyle = { padding: '5px', backgroundColor: '#2b2b43', color: '#fff', border: '1px solid #434651', borderRadius: '4px', cursor: 'pointer', fontSize: '10px' };
const cardStyle = { padding: '10px', backgroundColor: '#1e222d', borderRadius: '4px', marginBottom: '10px' };
const miniSelect = { background: '#090a0d', color: '#fff', border: 'none', fontSize: '11px', outline: 'none' };
const indicatorLabel = { fontSize: '11px', fontWeight: 'bold' };
const miniInput = { background: '#090a0d', color: '#fff', border: '1px solid #434651', padding: '2px', fontSize: '11px', width: '100%' };
const colorPicker = { width: '100%', height: '20px', border: 'none', background: 'none', cursor: 'pointer', padding: 0 };
const delBtn = { color: '#ef5350', border: 'none', background: 'none', cursor: 'pointer' };
const chartHeader = { padding: '10px 15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #2b2b43' };
const intervalBtn = { padding: '4px 8px', border: 'none', borderRadius: '2px', cursor: 'pointer', fontSize: '11px' };
const searchWrapper = { padding: '10px', borderBottom: '1px solid #2b2b43' };
const searchInput = { width: '100%', padding: '8px', background: '#090a0d', border: '1px solid #434651', color: '#fff', borderRadius: '4px', outline: 'none' };
const tickerTable = { width: '100%', borderCollapse: 'collapse' };
const tickerRow = { cursor: 'pointer', fontSize: '13px', borderBottom: '1px solid #2b2b43' };

export default App;