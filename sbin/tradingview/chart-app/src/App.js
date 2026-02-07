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

    // 거래 내역 관련 상태
    const [trades, setTrades] = useState([]);
    const [selectedTradeId, setSelectedTradeId] = useState(null);
    const [markers, setMarkers] = useState([]);
    const [showTradesPanel, setShowTradesPanel] = useState(false);

    // 사이드바 토글 상태
    const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
    const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false);

    const [indicators, setIndicators] = useState([
        { id: 1, category: 'ma', type: 'sma', period: '20', color: '#f2ff00', width: 2, active: true }
    ]);
    const [debouncedPeriods, setDebouncedPeriods] = useState({});

    const offsetRef = useRef(0);
    const isFetchingRef = useRef(false);

    useEffect(() => {
        const handler = setTimeout(() => {
            const periods = {};
            indicators.forEach(ind => {
                periods[ind.id] = ind.period;
                // vol의 volma period도 저장
                if (ind.category === 'vol' && ind.volmaActive) {
                    periods[`${ind.id}_volma`] = ind.volmaPeriod || '20';
                }
            });
            setDebouncedPeriods(periods);
        }, 400);
        return () => clearTimeout(handler);
    }, [indicators]); // indicators 변경 시 debounce 적용

    useEffect(() => {
        fetch(`http://localhost:8000/api/tickers`).then(res => res.json()).then(setTickers);
    }, []);

    // 거래 내역 목록 로드
    useEffect(() => {
        fetch(`http://localhost:8000/api/trades`)
            .then(res => res.json())
            .then(data => setTrades(data.trades || []))
            .catch(err => console.error('Failed to load trades:', err));
    }, []);

    // 선택된 거래의 마커 데이터 로드
    useEffect(() => {
        if (!selectedTradeId) {
            setMarkers([]);
            return;
        }
        fetch(`http://localhost:8000/api/trades/${selectedTradeId}`)
            .then(res => res.json())
            .then(data => {
                // 마커 데이터를 차트에 맞게 변환
                const chartMarkers = (data.markers || []).map(m => ({
                    time: m.time,
                    position: m.position,
                    color: m.color,
                    shape: m.shape,
                    text: m.text,
                    price: m.price,
                    details: m.details
                }));
                setMarkers(chartMarkers);
            })
            .catch(err => {
                console.error('Failed to load trade markers:', err);
                setMarkers([]);
            });
    }, [selectedTradeId]);

    const activeConfigs = useMemo(() => {
        const configs = [];
        indicators.filter(i => i.active).forEach(i => {
            const config = {
                ...i,
                period: debouncedPeriods[i.id] || i.period
            };
            configs.push(config);

            // vol이 활성화되어 있고 volma도 활성화되어 있으면 volma도 추가
            if (i.category === 'vol' && i.volmaActive) {
                configs.push({
                    id: i.id + '_volma',
                    category: 'volma',
                    type: 'volma',
                    period: debouncedPeriods[`${i.id}_volma`] || i.volmaPeriod || '20',
                    color: i.volmaColor || '#607D8B',
                    width: i.volmaWidth || 2,
                    active: true
                });
            }
        });
        return configs;
    }, [indicators, debouncedPeriods]);

    const fetchData = useCallback(async (isInitial = false) => {
        if (isFetchingRef.current || (!isInitial && !hasMore)) return;
        isFetchingRef.current = true;
        setIsFetching(true);

        if (isInitial) {
            offsetRef.current = 0;
            setHasMore(true);
        }

        const currentOffset = offsetRef.current;

        try {
            const params = new URLSearchParams();
            activeConfigs.forEach(i => params.append("configs", `${i.type}_${i.period}`));
            params.append("ticker", ticker);
            params.append("interval", interval);
            params.append("limit", 500);
            params.append("offset", currentOffset);

            const res = await fetch(`http://localhost:8000/api/ohlcv?${params.toString()}`);
            const newData = await res.json();

            if (!newData || newData.length === 0) {
                setHasMore(false);
            } else {
                setData(prev => {
                    const combined = isInitial ? newData : [...newData, ...prev];
                    // 시간 기준 중복 제거 및 정렬
                    return combined
                        .filter((v, i, a) => a.findIndex(t => t.time === v.time) === i)
                        .sort((a, b) => a.time - b.time);
                });
                offsetRef.current += 500;
                if (newData.length < 500) setHasMore(false);
            }
        } catch (error) { console.error(error); }
        finally { setIsFetching(false); isFetchingRef.current = false; }
    }, [ticker, interval, activeConfigs, hasMore]);

    useEffect(() => {
        fetchData(true);
    }, [ticker, interval, activeConfigs]);

    const addIndicator = (category) => {
        const id = Date.now();
        const base = { id, category, active: true, width: 2 };
        const presets = {
            'ma': { type: 'sma', period: '20', color: '#f2ff00' },
            'bb': { type: 'bollinger', period: '20', color: '#2196F3' },
            'ichi': { type: 'ichimoku', period: '9,26,52', color: '#9C27B0' },
            'rsi': { type: 'rsi', period: '14', color: '#E91E63' },
            'mfi': { type: 'mfi', period: '14', color: '#4CAF50' },
            'cci': { type: 'cci', period: '20', color: '#FF9800' },
            'vol': { type: 'volume', period: '0', color: '#607D8B', volmaActive: false, volmaPeriod: '20', volmaColor: '#607D8B', volmaWidth: 2 },
            'atr': { type: 'atr', period: '14', color: '#00BCD4' },
            'psar': { type: 'psar', period: '0', color: '#FFFFFF', width: 1 },
            'adx': { type: 'adx', period: '14', color: '#FFEB3B' },
            'dc': { type: 'donchian', period: '20', color: '#795548' }
        };

        const isOverlay = overlayCategories.includes(category);
        const isSeparated = separatedCategories.includes(category);

        // MA는 누를 때마다 계속 추가 (오버레이 지표)
        if (category === 'ma') {
            if (presets[category]) {
                // 오버레이 지표는 항상 앞에 추가
                setIndicators([...overlayIndicators, { ...base, ...presets[category] }, ...separatedIndicators]);
            }
            return;
        }

        // volma는 별도로 추가 불가 (vol에 포함)
        if (category === 'volma') {
            return;
        }

        // 나머지 보조지표는 토글 (있으면 제거, 없으면 추가)
        const exists = indicators.some(ind => ind.category === category);
        if (exists) {
            setIndicators(indicators.filter(ind => ind.category !== category));
        } else if (presets[category]) {
            if (isOverlay) {
                // 오버레이 지표는 오버레이 지표 섹션에 추가
                setIndicators([...overlayIndicators, { ...base, ...presets[category] }, ...separatedIndicators]);
            } else if (isSeparated) {
                // 분리 지표는 분리 지표 섹션에 추가
                setIndicators([...overlayIndicators, ...separatedIndicators, { ...base, ...presets[category] }]);
            }
        }
    };

    const updateIndicator = (id, field, value) => {
        setIndicators(indicators.map(ind => ind.id === id ? { ...ind, [field]: value } : ind));
    };

    // 오버레이 지표와 분리 지표 분리
    const overlayCategories = ['ma', 'bb', 'ichi', 'psar', 'dc'];
    const separatedCategories = ['rsi', 'mfi', 'cci', 'vol', 'atr', 'adx'];

    const overlayIndicators = useMemo(() => {
        return indicators.filter(ind => overlayCategories.includes(ind.category));
    }, [indicators]);

    const separatedIndicators = useMemo(() => {
        return indicators.filter(ind => separatedCategories.includes(ind.category));
    }, [indicators]);

    const moveIndicator = (id, direction) => {
        // 오버레이 지표는 순서 변경 불가
        const indicator = indicators.find(ind => ind.id === id);
        if (!indicator || overlayCategories.includes(indicator.category)) return;

        const separatedList = separatedIndicators;
        const index = separatedList.findIndex(ind => ind.id === id);
        if (index === -1) return;

        const newIndex = direction === 'up' ? index - 1 : index + 1;
        if (newIndex < 0 || newIndex >= separatedList.length) return;

        // 분리 지표만 재정렬
        const newSeparatedList = [...separatedList];
        [newSeparatedList[index], newSeparatedList[newIndex]] = [newSeparatedList[newIndex], newSeparatedList[index]];

        // 오버레이 지표 + 재정렬된 분리 지표로 합치기
        setIndicators([...overlayIndicators, ...newSeparatedList]);
    };

    return (
        <div style={containerStyle}>
            <div style={{
                ...leftSidebarStyle,
                width: isLeftSidebarOpen ? '250px' : '0px',
                padding: isLeftSidebarOpen ? '15px' : '0px',
                borderRight: isLeftSidebarOpen ? '1px solid #2b2b43' : 'none',
                overflow: 'hidden',
                transition: 'all 0.3s ease-in-out',
                opacity: isLeftSidebarOpen ? 1 : 0
            }}>
                {/* 오버레이 지표 섹션 */}
                <div style={{ marginBottom: '20px', minWidth: '220px' }}>
                    <h3 style={sidebarTitle}>오버레이 지표</h3>
                    <div style={addBtnGroup}>
                        {overlayCategories.map(cat => (
                            <button key={cat} onClick={() => addIndicator(cat)} style={addBtnStyle}>{cat.toUpperCase()}</button>
                        ))}
                    </div>
                    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                        {overlayIndicators.map((ind) => (
                            <IndicatorCard
                                key={ind.id}
                                ind={ind}
                                updateIndicator={updateIndicator}
                                onDelete={() => setIndicators(indicators.filter(i => i.id !== ind.id))}
                                canMove={false}
                            />
                        ))}
                    </div>
                </div>

                {/* 분리 지표 섹션 */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: '220px' }}>
                    <h3 style={sidebarTitle}>분리 지표</h3>
                    <div style={addBtnGroup}>
                        {separatedCategories.map(cat => (
                            <button key={cat} onClick={() => addIndicator(cat)} style={addBtnStyle}>{cat.toUpperCase()}</button>
                        ))}
                    </div>
                    <div style={{ overflowY: 'auto', flex: 1 }}>
                        {separatedIndicators.map((ind, idx) => (
                            <IndicatorCard
                                key={ind.id}
                                ind={ind}
                                updateIndicator={updateIndicator}
                                onDelete={() => setIndicators(indicators.filter(i => i.id !== ind.id))}
                                onMoveUp={() => moveIndicator(ind.id, 'up')}
                                onMoveDown={() => moveIndicator(ind.id, 'down')}
                                canMoveUp={idx > 0}
                                canMoveDown={idx < separatedIndicators.length - 1}
                                canMove={true}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <div style={chartAreaStyle}>
                <div style={chartHeader}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <button
                            onClick={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
                            style={{
                                background: 'transparent',
                                border: '1px solid #2b2b43',
                                color: '#848e9c',
                                cursor: 'pointer',
                                padding: '4px 8px',
                                borderRadius: '4px',
                                fontSize: '14px'
                            }}
                            title={isLeftSidebarOpen ? "지표 설정 닫기" : "지표 설정 열기"}
                        >
                            {isLeftSidebarOpen ? '◀' : '🛠️'}
                        </button>

                        <div style={{ fontWeight: 'bold', fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {ticker.replace("KRW-", "")}
                            <span style={{ fontSize: '12px', color: '#848e9c', fontWeight: 'normal' }}>{INTERVAL_OPTIONS.find(o => o.value === interval)?.label}</span>
                        </div>

                        <div style={{ display: 'flex', gap: '5px' }}>
                            {INTERVAL_OPTIONS.map(opt => (
                                <button
                                    key={opt.value}
                                    onClick={() => setInterval(opt.value)}
                                    style={{
                                        ...intervalBtn,
                                        backgroundColor: interval === opt.value ? '#2962ff' : 'transparent',
                                        color: interval === opt.value ? '#fff' : '#848e9c'
                                    }}
                                >
                                    {opt.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                        <button
                            onClick={() => setShowTradesPanel(!showTradesPanel)}
                            style={{
                                padding: '6px 12px',
                                backgroundColor: showTradesPanel ? '#2962ff' : '#2b2b43',
                                color: '#fff',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '13px',
                                fontWeight: 'bold',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px'
                            }}
                        >
                            📊 Trades {selectedTradeId && `(${selectedTradeId})`}
                        </button>
                        {selectedTradeId && (
                            <button
                                onClick={() => {
                                    setSelectedTradeId(null);
                                    setMarkers([]);
                                }}
                                style={{
                                    padding: '6px 12px',
                                    backgroundColor: 'transparent',
                                    color: '#ef5350',
                                    border: '1px solid #ef5350',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                    fontSize: '13px'
                                }}
                            >
                                ✕ Clear
                            </button>
                        )}

                        <button
                            onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)}
                            style={{
                                background: 'transparent',
                                border: '1px solid #2b2b43',
                                color: '#848e9c',
                                cursor: 'pointer',
                                padding: '4px 8px',
                                borderRadius: '4px',
                                fontSize: '14px'
                            }}
                            title={isRightSidebarOpen ? "검색창 닫기" : "검색창 열기"}
                        >
                            {isRightSidebarOpen ? '▶' : '🔍'}
                        </button>
                    </div>
                </div>

                {/* 거래 선택 패널 */}
                {showTradesPanel && (
                    <div style={{
                        padding: '20px',
                        borderBottom: '1px solid #2b2b43',
                        backgroundColor: '#131722',
                        maxHeight: '500px',
                        overflowY: 'auto',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                        position: 'absolute',
                        top: '50px',
                        left: 0,
                        right: 0,
                        zIndex: 20
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                            <div style={{ fontSize: '16px', color: '#d1d4dc', fontWeight: 'bold' }}>
                                📊 거래 내역 선택 ({trades.length}개)
                            </div>
                            <button
                                onClick={() => setShowTradesPanel(false)}
                                style={{ background: 'none', border: 'none', color: '#848e9c', cursor: 'pointer', fontSize: '16px' }}
                            >✕</button>
                        </div>

                        {trades.length === 0 ? (
                            <div style={{ color: '#848e9c', fontSize: '14px', padding: '40px', textAlign: 'center', backgroundColor: '#1e222d', borderRadius: '8px' }}>
                                거래 데이터가 없습니다.<br />
                                <div style={{ marginTop: '10px', fontSize: '12px' }}>
                                    백테스트 실행 시 <code style={{ color: '#4caf50', backgroundColor: '#2b2b43', padding: '2px 6px', borderRadius: '4px' }}>--export_trades var/trades/</code> 옵션을 사용하세요.
                                </div>
                            </div>
                        ) : (
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '12px' }}>
                                {trades.map(trade => (
                                    <div
                                        key={trade.id}
                                        onClick={() => {
                                            setSelectedTradeId(trade.id);
                                            setShowTradesPanel(false);
                                            // 해당 티커로 자동 이동
                                            if (trade.ticker) {
                                                setTicker(trade.ticker);
                                            }
                                        }}
                                        style={{
                                            display: 'flex',
                                            flexDirection: 'column',
                                            padding: '15px',
                                            backgroundColor: selectedTradeId === trade.id ? 'rgba(41, 98, 255, 0.15)' : '#1e222d',
                                            border: selectedTradeId === trade.id ? '1px solid #2962ff' : '1px solid #2b2b43',
                                            borderRadius: '8px',
                                            cursor: 'pointer',
                                            transition: 'all 0.2s',
                                            position: 'relative',
                                            overflow: 'hidden'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.transform = 'translateY(-2px)';
                                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
                                            if (selectedTradeId !== trade.id) e.currentTarget.style.borderColor = '#4caf50';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.transform = 'translateY(0)';
                                            e.currentTarget.style.boxShadow = 'none';
                                            if (selectedTradeId !== trade.id) e.currentTarget.style.borderColor = '#2b2b43';
                                        }}
                                    >
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
                                            <div>
                                                <div style={{ fontWeight: 'bold', fontSize: '15px', color: '#d1d4dc' }}>{trade.ticker || 'Unknown'}</div>
                                                <div style={{ fontSize: '11px', color: '#848e9c', marginTop: '2px' }}>{trade.interval}</div>
                                            </div>
                                            <div style={{ textAlign: 'right' }}>
                                                <div style={{
                                                    color: trade.total_pnl > 0 ? '#4caf50' : '#ef5350',
                                                    fontWeight: 'bold',
                                                    fontSize: '16px'
                                                }}>
                                                    {(trade.total_pnl * 100).toFixed(2)}%
                                                </div>
                                                <div style={{ fontSize: '11px', color: '#848e9c' }}>PnL</div>
                                            </div>
                                        </div>

                                        <div style={{ fontSize: '12px', color: '#b2b5be', marginBottom: '12px', lineHeight: '1.4' }}>
                                            <div><span style={{ color: '#848e9c' }}>Buy:</span> {trade.buy_strategy}</div>
                                            <div><span style={{ color: '#848e9c' }}>Sell:</span> {trade.sell_strategy}</div>
                                        </div>

                                        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #2b2b43', paddingTop: '10px', fontSize: '12px' }}>
                                            <div style={{ textAlign: 'center' }}>
                                                <div style={{ color: '#d1d4dc', fontWeight: 'bold' }}>{trade.trade_count}</div>
                                                <div style={{ color: '#848e9c', fontSize: '10px' }}>Trades</div>
                                            </div>
                                            <div style={{ textAlign: 'center' }}>
                                                <div style={{ color: '#d1d4dc', fontWeight: 'bold' }}>{(trade.win_ratio * 100).toFixed(1)}%</div>
                                                <div style={{ color: '#848e9c', fontSize: '10px' }}>Win Rate</div>
                                            </div>
                                            <div style={{ textAlign: 'center' }}>
                                                <div style={{ color: '#ef5350', fontWeight: 'bold' }}>{(trade.mdd * 100).toFixed(1)}%</div>
                                                <div style={{ color: '#848e9c', fontSize: '10px' }}>MDD</div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                <div style={{ flex: 1, position: 'relative' }}>
                    <ChartComponent data={data} configs={activeConfigs} markers={markers} onLoadMore={() => fetchData(false)} />
                </div>
            </div>

            <div style={{
                ...rightSidebarStyle,
                width: isRightSidebarOpen ? '260px' : '0px',
                borderLeft: isRightSidebarOpen ? '1px solid #2b2b43' : 'none',
                overflow: 'hidden',
                transition: 'all 0.3s ease-in-out',
                opacity: isRightSidebarOpen ? 1 : 0
            }}>
                <div style={{ ...searchWrapper, minWidth: '240px' }}><input type="text" placeholder="심볼 검색" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} style={searchInput} /></div>
                <div style={{ flex: 1, overflowY: 'auto', minWidth: '240px' }}>
                    <table style={tickerTable}>
                        <tbody>{tickers.filter(t => t.replace("KRW-", "").toLowerCase().includes(searchTerm.toLowerCase())).map(t => (
                            <tr key={t} onClick={() => setTicker(t)} style={{ ...tickerRow, backgroundColor: ticker === t ? '#2b2b43' : 'transparent' }}>
                                <td style={{ padding: '10px' }}><div style={{ fontWeight: 'bold', fontSize: '13px' }}>{t.replace("KRW-", "")}</div><div style={{ fontSize: '11px', color: '#848e9c' }}>KRW/{t.replace("KRW-", "")}</div></td>
                            </tr>
                        ))}</tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

const containerStyle = { display: 'flex', backgroundColor: '#090a0d', height: '100vh', width: '100vw', color: '#d1d4dc', overflow: 'hidden' };
const leftSidebarStyle = { width: '240px', backgroundColor: '#131722', borderRight: '1px solid #2b2b43', padding: '15px', display: 'flex', flexDirection: 'column' };
const chartAreaStyle = { flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #2b2b43', position: 'relative' };
const rightSidebarStyle = { width: '260px', backgroundColor: '#131722', display: 'flex', flexDirection: 'column' };
const sidebarTitle = { fontSize: '14px', marginBottom: '15px', color: '#848e9c' };
const addBtnGroup = { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px', marginBottom: '15px' };
const addBtnStyle = { padding: '6px', backgroundColor: '#2b2b43', color: '#fff', border: '1px solid #434651', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' };
const cardStyle = { padding: '10px', backgroundColor: '#1e222d', borderRadius: '4px', marginBottom: '10px', border: '1px solid #2b2b43' };
const miniSelect = { background: '#090a0d', color: '#fff', border: 'none', fontSize: '11px', outline: 'none' };
const indicatorLabel = { fontSize: '11px', fontWeight: 'bold', color: '#fff' };
const miniInput = { background: '#090a0d', color: '#fff', border: '1px solid #434651', padding: '3px', fontSize: '11px', width: '100%' };
const colorPicker = { width: '100%', height: '22px', border: 'none', background: 'none', cursor: 'pointer' };
const delBtn = { color: '#ef5350', border: 'none', background: 'none', cursor: 'pointer', fontSize: '14px' };
const moveBtn = { color: '#848e9c', border: 'none', background: 'none', cursor: 'pointer', fontSize: '12px', padding: '2px 4px' };
const chartHeader = { padding: '10px 15px', borderBottom: '1px solid #2b2b43', display: 'flex', justifyContent: 'space-between', alignItems: 'center' };
const intervalBtn = { padding: '4px 8px', border: 'none', borderRadius: '2px', cursor: 'pointer', fontSize: '11px' };
const searchWrapper = { padding: '10px', borderBottom: '1px solid #2b2b43' };
const searchInput = { width: '100%', padding: '8px', backgroundColor: '#090a0d', border: '1px solid #434651', color: '#fff', borderRadius: '4px', outline: 'none' };
const tickerTable = { width: '100%', borderCollapse: 'collapse' };
const tableHead = { backgroundColor: '#1e222d', color: '#848e9c', fontSize: '11px' };
const tickerRow = { borderBottom: '1px solid #2b2b43', cursor: 'pointer' };

const IndicatorCard = ({ ind, updateIndicator, onDelete, onMoveUp, onMoveDown, canMoveUp, canMoveDown, canMove }) => {
    return (
        <div style={{ ...cardStyle, borderLeft: `4px solid ${ind.color}`, opacity: ind.active ? 1 : 0.6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <input type="checkbox" checked={ind.active} onChange={e => updateIndicator(ind.id, 'active', e.target.checked)} />
                    {ind.category === 'ma' ? (
                        <select value={ind.type} onChange={e => updateIndicator(ind.id, 'type', e.target.value)} style={miniSelect}>
                            <option value="sma">SMA</option><option value="ema">EMA</option><option value="wma">WMA</option>
                        </select>
                    ) : <span style={indicatorLabel}>{ind.type.toUpperCase()}</span>}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {canMove && (
                        <>
                            <button
                                onClick={onMoveUp}
                                disabled={!canMoveUp}
                                style={{ ...moveBtn, opacity: canMoveUp ? 1 : 0.3, cursor: canMoveUp ? 'pointer' : 'not-allowed' }}
                                title="위로 이동"
                            >
                                ↑
                            </button>
                            <button
                                onClick={onMoveDown}
                                disabled={!canMoveDown}
                                style={{ ...moveBtn, opacity: canMoveDown ? 1 : 0.3, cursor: canMoveDown ? 'pointer' : 'not-allowed' }}
                                title="아래로 이동"
                            >
                                ↓
                            </button>
                        </>
                    )}
                    <button onClick={onDelete} style={delBtn}>✕</button>
                </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr 1fr', gap: '5px' }}>
                {ind.category === 'vol' ? (
                    <>
                        <div />{/* volume은 period, width, color 모두 숨김 */}
                        <div />
                        <div />
                    </>
                ) : (
                    <>
                        <input type="text" value={ind.period} onChange={e => updateIndicator(ind.id, 'period', e.target.value)} style={miniInput} />
                        <select value={ind.width} onChange={e => updateIndicator(ind.id, 'width', parseInt(e.target.value))} style={miniSelect}>
                            {[1, 2, 3, 4, 5].map(w => <option key={w} value={w}>{w}px</option>)}
                        </select>
                        <input type="color" value={ind.color} onChange={e => updateIndicator(ind.id, 'color', e.target.value)} style={colorPicker} />
                    </>
                )}
            </div>
            {ind.category === 'vol' && (
                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #2b2b43' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                        <input
                            type="checkbox"
                            checked={ind.volmaActive || false}
                            onChange={e => updateIndicator(ind.id, 'volmaActive', e.target.checked)}
                        />
                        <span style={{ fontSize: '11px', color: '#d1d4dc' }}>VOLMA</span>
                    </div>
                    {ind.volmaActive && (
                        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr 1fr', gap: '5px' }}>
                            <input
                                type="text"
                                value={ind.volmaPeriod || '20'}
                                onChange={e => updateIndicator(ind.id, 'volmaPeriod', e.target.value)}
                                style={miniInput}
                                placeholder="Period"
                            />
                            <select
                                value={ind.volmaWidth || 2}
                                onChange={e => updateIndicator(ind.id, 'volmaWidth', parseInt(e.target.value))}
                                style={miniSelect}
                            >
                                {[1, 2, 3, 4, 5].map(w => <option key={w} value={w}>{w}px</option>)}
                            </select>
                            <input
                                type="color"
                                value={ind.volmaColor || '#607D8B'}
                                onChange={e => updateIndicator(ind.id, 'volmaColor', e.target.value)}
                                style={colorPicker}
                            />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default App;