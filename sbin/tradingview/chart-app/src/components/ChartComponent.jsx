import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';

const ChartComponent = ({ data, configs, onLoadMore }) => {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candleSeriesRef = useRef();
    // 각 보조지표 라인/시리즈와 레이블 정보를 함께 저장
    const indicatorSeriesRef = useRef([]); // [{ series, label }]
    const firstTimeRef = useRef(null);
    const [hoverInfo, setHoverInfo] = useState(null);
    
    // 부모의 onLoadMore를 최신 상태로 참조하기 위함
    const onLoadMoreRef = useRef(onLoadMore);
    useEffect(() => { onLoadMoreRef.current = onLoadMore; }, [onLoadMore]);

    useEffect(() => {
        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#090a0d' }, textColor: '#d1d4dc' },
            grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: { borderVisible: false, timeVisible: true, rightOffset: 5, minBarSpacing: 0.5 },
        });

        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#e21010', downColor: '#0051ff', borderVisible: false,
            wickUpColor: '#e21010', wickDownColor: '#0051ff',
            priceLineVisible: false, lastValueVisible: false
        });

        candleSeriesRef.current = candleSeries;
        chartRef.current = chart;

        // [핵심] 스크롤이 왼쪽 끝(과거)에 닿으면 데이터 추가 로드
        const handleVisibleRangeChange = () => {
            const range = chart.timeScale().getVisibleRange();
            if (!range || !firstTimeRef.current) return;
            
            // 현재 보이는 첫 번째 봉이 우리가 가진 가장 오래된 봉 근처일 때
            if (range.from < firstTimeRef.current + 10) {
                onLoadMoreRef.current();
            }
        };
        chart.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRangeChange);

        // 크로스헤어 이동 시 OHLC + 보조지표 값을 수집
        const handleCrosshairMove = (param) => {
            if (!param || !param.time || !param.point || !chartRef.current) {
                setHoverInfo(null);
                return;
            }

            // 차트 영역 밖이면 표시하지 않음
            if (param.point.x < 0 || param.point.y < 0) {
                setHoverInfo(null);
                return;
            }

            const seriesDataMap = param.seriesData;
            const candleData = seriesDataMap.get(candleSeriesRef.current);

            const indicators = [];
            const pricesMap = param.seriesPrices;

            indicatorSeriesRef.current.forEach(({ series, label }) => {
                if (!series || !label) return;
                let value = undefined;
                if (pricesMap && typeof pricesMap.get === 'function') {
                    value = pricesMap.get(series);
                } else if (seriesDataMap && typeof seriesDataMap.get === 'function') {
                    const d = seriesDataMap.get(series);
                    if (d && typeof d.value === 'number') value = d.value;
                }
                if (value != null && !Number.isNaN(value)) {
                    indicators.push({ label, value });
                }
            });

            // time 문자열 포맷 (간단하게 처리)
            let timeLabel = '';
            const t = param.time;
            if (typeof t === 'number') {
                // lightweight-charts 기본 Unix timestamp(sec) 가정
                timeLabel = new Date(t * 1000).toLocaleString('ko-KR');
            } else if (t && typeof t === 'object' && 'year' in t) {
                timeLabel = `${t.year}-${String(t.month).padStart(2, '0')}-${String(t.day).padStart(2, '0')}`;
            }

            setHoverInfo({
                time: timeLabel,
                candle: candleData || null,
                indicators,
            });
        };
        chart.subscribeCrosshairMove(handleCrosshairMove);

        const handleResize = () => chartRef.current && chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth, 
            height: chartContainerRef.current.clientHeight 
        });

        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
            chart.timeScale().unsubscribeVisibleTimeRangeChange(handleVisibleRangeChange);
            chart.unsubscribeCrosshairMove(handleCrosshairMove);
            chart.remove();
        };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0 || !chartRef.current) return;
        
        // 우리가 가진 가장 과거 시간 저장
        firstTimeRef.current = data[0].time;
        
        // 캔들 데이터 세팅 (null/NaN 값 필터링)
        const validCandleData = data.filter(d => 
            d.open != null && !Number.isNaN(d.open) &&
            d.high != null && !Number.isNaN(d.high) &&
            d.low != null && !Number.isNaN(d.low) &&
            d.close != null && !Number.isNaN(d.close)
        );
        candleSeriesRef.current.setData(validCandleData);

        // 이전 지표 청소
        indicatorSeriesRef.current.forEach(({ series }) => chartRef.current.removeSeries(series));
        indicatorSeriesRef.current = [];

        // 보조지표 그리기
        configs.forEach(conf => {
            const baseOptions = { 
                color: conf.color, 
                lineWidth: conf.width || 2, 
                priceLineVisible: false, 
                lastValueVisible: false 
            };
            
            if (['sma', 'ema', 'wma'].includes(conf.type)) {
                const key = conf.type.toUpperCase() + conf.period;
                const lineData = data
                    .filter(d => d.mas && d.mas[key] != null && !Number.isNaN(d.mas[key]))
                    .map(d => ({ time: d.time, value: d.mas[key] }));
                if (lineData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, baseOptions);
                    s.setData(lineData);
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'bollinger') {
                const key = `BB${conf.period}`;
                const bbData = data.filter(d => d.bbs && d.bbs[key]);
                if (bbData.length > 0) {
                    const up = chartRef.current.addSeries(LineSeries, baseOptions);
                    const lo = chartRef.current.addSeries(LineSeries, baseOptions);
                    const upData = bbData
                        .filter(d => d.bbs[key].up != null && !Number.isNaN(d.bbs[key].up))
                        .map(d => ({ time: d.time, value: d.bbs[key].up }));
                    const loData = bbData
                        .filter(d => d.bbs[key].dn != null && !Number.isNaN(d.bbs[key].dn))
                        .map(d => ({ time: d.time, value: d.bbs[key].dn }));
                    up.setData(upData);
                    lo.setData(loData);
                    indicatorSeriesRef.current.push(
                        { series: up, label: `${key}_UP` },
                        { series: lo, label: `${key}_DN` }
                    );
                }
            } else if (conf.type === 'ichimoku') {
                const key = `ICHI${conf.period.replace(/,/g, '_')}`;
                const ichiData = data.filter(d => d.ichis && d.ichis[key]);
                if (ichiData.length > 0) {
                    const sa = chartRef.current.addSeries(LineSeries, baseOptions);
                    const sb = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: conf.color + '80' });
                    const saData = ichiData
                        .filter(d => d.ichis[key].sa != null && !Number.isNaN(d.ichis[key].sa))
                        .map(d => ({ time: d.time, value: d.ichis[key].sa }));
                    const sbData = ichiData
                        .filter(d => d.ichis[key].sb != null && !Number.isNaN(d.ichis[key].sb))
                        .map(d => ({ time: d.time, value: d.ichis[key].sb }));
                    sa.setData(saData);
                    sb.setData(sbData);
                    indicatorSeriesRef.current.push(
                        { series: sa, label: `${key}_SA` },
                        { series: sb, label: `${key}_SB` }
                    );
                }
            } else if (['rsi', 'mfi', 'cci'].includes(conf.type)) {
                const key = `${conf.type.toUpperCase()}${conf.period}`;
                const paneId = `pane_${key}`;
                const oscData = data
                    .filter(d => d.oscillators && d.oscillators[key] != null && !Number.isNaN(d.oscillators[key]))
                    .map(d => ({ time: d.time, value: d.oscillators[key] }));
                if (oscData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(oscData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: { top: 0.8, bottom: 0.05 },
                        position: 'left'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'volume') {
                const volData = data.filter(d => d.volume != null).map(d => ({
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? 'rgba(226, 16, 16, 0.6)' : 'rgba(0, 81, 255, 0.6)'
                }));
                if (volData.length > 0) {
                    const s = chartRef.current.addSeries(HistogramSeries, { 
                        priceFormat: { type: 'volume' }, 
                        priceScaleId: 'vol',
                        priceLineVisible: false,
                        lastValueVisible: false
                    });
                    s.setData(volData);
                    // volume 시리즈 추가 후 price scale 설정
                    chartRef.current.priceScale('vol').applyOptions({ 
                        scaleMargins: { top: 0.8, bottom: 0 },
                        position: 'left'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: 'VOL' });
                }
            } else if (conf.type === 'vol_sma' || conf.type === 'volma') {
                const key = `VOLMA${conf.period}`;
                const volmaData = data
                    .filter(d => d.volumes && d.volumes[key] != null && !Number.isNaN(d.volumes[key]))
                    .map(d => ({ time: d.time, value: d.volumes[key] }));
                if (volmaData.length > 0) {
                    // volma만 켠 경우에도 vol priceScale이 "항상" 존재하도록 보이지 않는 볼륨 히스토그램을 먼저 추가
                    // (lightweight-charts는 series가 붙어야 해당 priceScaleId가 생성됨)
                    const hiddenVolData = data
                        .filter(d => d.volume != null)
                        .map(d => ({
                            time: d.time,
                            value: d.volume,
                            color: 'rgba(0,0,0,0)' // 완전 투명
                        }));
                    if (hiddenVolData.length > 0) {
                        const hiddenVol = chartRef.current.addSeries(HistogramSeries, {
                            priceFormat: { type: 'volume' },
                            priceScaleId: 'vol',
                            priceLineVisible: false,
                            lastValueVisible: false
                        });
                        hiddenVol.setData(hiddenVolData);
                        chartRef.current.priceScale('vol').applyOptions({ 
                            scaleMargins: { top: 0.8, bottom: 0 },
                            position: 'left'
                        });
                        indicatorSeriesRef.current.push({ series: hiddenVol, label: 'VOL(hidden)' });
                    }

                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: 'vol' });
                    s.setData(volmaData);
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'atr') {
                const key = `ATR${conf.period}`;
                const paneId = `pane_${key}`;
                const atrData = data
                    .filter(d => d.atrs && d.atrs[key] != null && !Number.isNaN(d.atrs[key]))
                    .map(d => ({ time: d.time, value: d.atrs[key] }));
                if (atrData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(atrData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: { top: 0.8, bottom: 0.05 },
                        position: 'left'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'psar') {
                const psarData = data
                    .filter(d => d.psars && d.psars['PSAR'] != null && !Number.isNaN(d.psars['PSAR']))
                    .map(d => ({ time: d.time, value: d.psars['PSAR'] }));
                if (psarData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { 
                        ...baseOptions, 
                        lineStyle: 2, 
                        lineWidth: 0, 
                        markerType: 'circle',
                        markerSize: 3
                    });
                    s.setData(psarData);
                    indicatorSeriesRef.current.push({ series: s, label: 'PSAR' });
                }
            } else if (conf.type === 'adx') {
                const key = `ADX${conf.period}`;
                const paneId = `pane_${key}`;
                const adxData = data.filter(d => d.adxs && d.adxs[key]);
                if (adxData.length > 0) {
                    const sAdx = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    const sPlus = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: '#4CAF50', lineWidth: 1, priceScaleId: paneId });
                    const sMinus = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: '#F44336', lineWidth: 1, priceScaleId: paneId });
                    const adxLineData = adxData
                        .filter(d => d.adxs[key].adx != null && !Number.isNaN(d.adxs[key].adx))
                        .map(d => ({ time: d.time, value: d.adxs[key].adx }));
                    const plusLineData = adxData
                        .filter(d => d.adxs[key].plus != null && !Number.isNaN(d.adxs[key].plus))
                        .map(d => ({ time: d.time, value: d.adxs[key].plus }));
                    const minusLineData = adxData
                        .filter(d => d.adxs[key].minus != null && !Number.isNaN(d.adxs[key].minus))
                        .map(d => ({ time: d.time, value: d.adxs[key].minus }));
                    sAdx.setData(adxLineData);
                    sPlus.setData(plusLineData);
                    sMinus.setData(minusLineData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: { top: 0.75, bottom: 0.05 },
                        position: 'left'
                    });
                    indicatorSeriesRef.current.push(
                        { series: sAdx, label: key },
                        { series: sPlus, label: `${key}_PLUS` },
                        { series: sMinus, label: `${key}_MINUS` },
                    );
                }
            } else if (conf.type === 'donchian') {
                const key = `DC${conf.period}`;
                const dcData = data.filter(d => d.donchians && d.donchians[key]);
                if (dcData.length > 0) {
                    const up = chartRef.current.addSeries(LineSeries, baseOptions);
                    const lo = chartRef.current.addSeries(LineSeries, baseOptions);
                    const upData = dcData
                        .filter(d => d.donchians[key].up != null && !Number.isNaN(d.donchians[key].up))
                        .map(d => ({ time: d.time, value: d.donchians[key].up }));
                    const loData = dcData
                        .filter(d => d.donchians[key].lo != null && !Number.isNaN(d.donchians[key].lo))
                        .map(d => ({ time: d.time, value: d.donchians[key].lo }));
                    up.setData(upData);
                    lo.setData(loData);
                    indicatorSeriesRef.current.push(
                        { series: up, label: `${key}_UP` },
                        { series: lo, label: `${key}_LO` },
                    );
                }
            }
        });
    }, [data, configs]);

    return (
        <div
            ref={chartContainerRef}
            style={{ width: '100%', height: '100%', position: 'relative', fontFamily: 'Roboto, -apple-system, BlinkMacSystemFont, system-ui, sans-serif' }}
        >
            {hoverInfo && (
                <div
                    style={{
                        position: 'absolute',
                        top: 8,
                        left: 8,
                        padding: '6px 8px',
                        backgroundColor: 'rgba(9,10,13,0.9)',
                        border: '1px solid #2b2b43',
                        borderRadius: 4,
                        fontSize: 11,
                        color: '#d1d4dc',
                        pointerEvents: 'none',
                        maxWidth: '60%',
                        zIndex: 10,
                    }}
                >
                    {hoverInfo.time && (
                        <div style={{ marginBottom: 2, color: '#848e9c' }}>{hoverInfo.time}</div>
                    )}
                    {hoverInfo.candle && (
                        <div style={{ marginBottom: 2 }}>
                            <span style={{ color: '#f2ff00', fontWeight: 'bold', marginRight: 4 }}>OHLC</span>
                            <span>O {hoverInfo.candle.open?.toFixed?.(2)} </span>
                            <span>H {hoverInfo.candle.high?.toFixed?.(2)} </span>
                            <span>L {hoverInfo.candle.low?.toFixed?.(2)} </span>
                            <span>C {hoverInfo.candle.close?.toFixed?.(2)}</span>
                        </div>
                    )}
                    {hoverInfo.indicators && hoverInfo.indicators.length > 0 && (
                        <div>
                            {Array.from({ length: Math.ceil(hoverInfo.indicators.length / 5) }, (_, rowIdx) => {
                                const startIdx = rowIdx * 5;
                                const endIdx = startIdx + 5;
                                const rowIndicators = hoverInfo.indicators.slice(startIdx, endIdx);
                                return (
                                    <div
                                        key={rowIdx}
                                        style={{
                                            display: 'flex',
                                            flexWrap: 'nowrap',
                                            gap: 4,
                                            marginBottom: rowIdx < Math.ceil(hoverInfo.indicators.length / 5) - 1 ? 2 : 0,
                                        }}
                                    >
                                        {rowIndicators.map((ind, idx) => (
                                            <div key={startIdx + idx} style={{ whiteSpace: 'nowrap' }}>
                                                <span style={{ color: '#4cafef', marginRight: 3 }}>{ind.label}:</span>
                                                <span>{Number(ind.value).toFixed(2)}</span>
                                            </div>
                                        ))}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ChartComponent;