import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';

const ChartComponent = ({ data, configs, onLoadMore }) => {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candleSeriesRef = useRef();
    const indicatorSeriesRef = useRef([]); 
    const firstTimeRef = useRef(null);
    const onLoadMoreRef = useRef(onLoadMore);

    useEffect(() => { onLoadMoreRef.current = onLoadMore; }, [onLoadMore]);

    // 키 매핑 헬퍼
    const getKey = (conf) => {
        const p = conf.period.replace(/,/g, '_');
        if (conf.type === 'bollinger') return `BB${p}`;
        if (conf.type === 'donchian') return `DC${p}`;
        if (conf.type === 'ichimoku') return `ICHI${p}`;
        if (conf.type === 'psar') return `PSAR`;
        return `${conf.type.toUpperCase()}${p}`;
    };

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // 차트 생성
        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#090a0d' }, textColor: '#d1d4dc' },
            grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: { borderVisible: false, timeVisible: true, rightOffset: 5 },
            crosshair: { mode: 1 },
        });

        // 캔들 시리즈
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#e21010', downColor: '#0051ff', borderVisible: false,
            wickUpColor: '#e21010', wickDownColor: '#0051ff',
            priceLineVisible: false, // 현재가 점선 제거
        });

        candleSeriesRef.current = candleSeries;
        chartRef.current = chart;

        // 무한 스크롤
        chart.timeScale().subscribeVisibleTimeRangeChange(() => {
            const range = chart.timeScale().getVisibleRange();
            if (range && firstTimeRef.current && range.from < firstTimeRef.current + 10) {
                onLoadMoreRef.current();
            }
        });

        const handleResize = () => {
            if (chartRef.current) {
                chartRef.current.applyOptions({ 
                    width: chartContainerRef.current.clientWidth, 
                    height: chartContainerRef.current.clientHeight 
                });
            }
        };

        window.addEventListener('resize', handleResize);
        return () => { window.removeEventListener('resize', handleResize); chart.remove(); };
    }, []);

    // 데이터 및 지표 업데이트
    useEffect(() => {
        if (!data || data.length === 0 || !chartRef.current) return;
        
        firstTimeRef.current = data[0].time;
        candleSeriesRef.current.setData(data);

        // 기존 지표 제거
        indicatorSeriesRef.current.forEach(s => {
            try { chartRef.current.removeSeries(s); } catch(e){}
        });
        indicatorSeriesRef.current = [];

        // 거래량 히스토그램
        const volSeries = chartRef.current.addSeries(HistogramSeries, {
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: '', 
        });
        volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
        
        const volData = data.map(d => ({
            time: d.time, value: d.volume,
            color: d.close >= d.open ? '#e2101044' : '#0051ff44'
        }));
        volSeries.setData(volData);
        indicatorSeriesRef.current.push(volSeries);

        // --- 사용자 설정 지표 그리기 ---
        configs.forEach(conf => {
            const key = getKey(conf);
            const baseOpt = { 
                color: conf.color, 
                lineWidth: conf.width || 2, 
                priceLineVisible: false, 
                lastValueVisible: true,
                crosshairMarkerVisible: true,
            };

            const validData = data.filter(d => d.inds && d.inds[key] !== undefined && d.inds[key] !== null);

            // 1. 단일 라인 오버레이 (SMA, EMA, WMA, PSAR)
            if (['sma', 'ema', 'wma', 'psar'].includes(conf.type)) {
                const s = chartRef.current.addSeries(LineSeries, {
                    ...baseOpt,
                    ...(conf.type === 'psar' ? { lineStyle: 2, lineWidth: 0, markerType: 'circle' } : {}) 
                });
                s.setData(validData.map(d => ({ time: d.time, value: d.inds[key] })));
                indicatorSeriesRef.current.push(s);
            }
            
            // 2. 밴드형 오버레이 - [수정됨] 볼린저와 일목균형표 분리 처리
            else if (conf.type === 'bollinger' || conf.type === 'donchian') {
                const up = chartRef.current.addSeries(LineSeries, baseOpt);
                const lo = chartRef.current.addSeries(LineSeries, baseOpt);
                
                // Bollinger/Donchian은 .up, .lo 필드를 가짐
                up.setData(validData.map(d => ({ time: d.time, value: d.inds[key].up })));
                lo.setData(validData.map(d => ({ time: d.time, value: d.inds[key].lo })));
                
                indicatorSeriesRef.current.push(up, lo);
            }
            else if (conf.type === 'ichimoku') {
                const sa = chartRef.current.addSeries(LineSeries, { ...baseOpt, title: 'Span A' });
                const sb = chartRef.current.addSeries(LineSeries, { ...baseOpt, color: conf.color + '88', title: 'Span B' }); // 반투명
                
                // Ichimoku는 .sa, .sb 필드를 가짐
                sa.setData(validData.map(d => ({ time: d.time, value: d.inds[key].sa })));
                sb.setData(validData.map(d => ({ time: d.time, value: d.inds[key].sb })));
                
                indicatorSeriesRef.current.push(sa, sb);
            }
            
            // 3. 별도 Pane 오실레이터 (RSI, CCI, ADX 등) - [수정됨] Pane 설정 순서 변경
            else if (['rsi', 'cci', 'mfi', 'atr', 'adx', 'volma'].includes(conf.type)) {
                const paneId = `pane_${key}`; // 고유 Pane ID
                
                if (conf.type === 'adx') {
                    // 시리즈 먼저 생성
                    const sAdx = chartRef.current.addSeries(LineSeries, { ...baseOpt, priceScaleId: paneId });
                    const sPlus = chartRef.current.addSeries(LineSeries, { ...baseOpt, color: '#4CAF50', priceScaleId: paneId, lineWidth: 1 });
                    const sMinus = chartRef.current.addSeries(LineSeries, { ...baseOpt, color: '#F44336', priceScaleId: paneId, lineWidth: 1 });
                    
                    // 데이터 주입
                    sAdx.setData(validData.map(d => ({ time: d.time, value: d.inds[key].adx })));
                    sPlus.setData(validData.map(d => ({ time: d.time, value: d.inds[key].plus })));
                    sMinus.setData(validData.map(d => ({ time: d.time, value: d.inds[key].minus })));
                    
                    indicatorSeriesRef.current.push(sAdx, sPlus, sMinus);
                } else {
                    // 시리즈 먼저 생성
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOpt, priceScaleId: paneId });
                    
                    // 데이터 주입
                    s.setData(validData.map(d => ({ time: d.time, value: d.inds[key] })));
                    indicatorSeriesRef.current.push(s);
                }

                // [중요] 시리즈가 생성된 **후에** 스케일 옵션을 적용해야 에러가 안 남
                chartRef.current.priceScale(paneId).applyOptions({
                    scaleMargins: { top: 0.75, bottom: 0.05 }, // 화면 하단 25% 사용
                    visible: true,
                    borderVisible: true,
                    borderColor: '#2b2b43',
                });
            }
        });

    }, [data, configs]);

    return <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />;
};

export default ChartComponent;