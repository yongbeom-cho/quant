import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';

const ChartComponent = ({ data, configs, onLoadMore }) => {
    const chartContainerRef = useRef();
    const chartRef = useRef();
    const candleSeriesRef = useRef();
    const indicatorSeriesRef = useRef([]);
    const firstTimeRef = useRef(null);
    
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
        chart.timeScale().subscribeVisibleTimeRangeChange(() => {
            const range = chart.timeScale().getVisibleRange();
            if (!range || !firstTimeRef.current) return;
            
            // 현재 보이는 첫 번째 봉이 우리가 가진 가장 오래된 봉 근처일 때
            if (range.from < firstTimeRef.current + 10) {
                onLoadMoreRef.current();
            }
        });

        const handleResize = () => chartRef.current && chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth, 
            height: chartContainerRef.current.clientHeight 
        });

        window.addEventListener('resize', handleResize);
        return () => { window.removeEventListener('resize', handleResize); chart.remove(); };
    }, []);

    useEffect(() => {
        if (!data || data.length === 0 || !chartRef.current) return;
        
        // 우리가 가진 가장 과거 시간 저장
        firstTimeRef.current = data[0].time;
        
        // 캔들 데이터 세팅
        candleSeriesRef.current.setData(data);

        // 이전 지표 청소
        indicatorSeriesRef.current.forEach(s => chartRef.current.removeSeries(s));
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
                const lineData = data.filter(d => d.mas && d.mas[key]).map(d => ({ time: d.time, value: d.mas[key] }));
                if (lineData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, baseOptions);
                    s.setData(lineData);
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'bollinger') {
                const key = `BB${conf.period}`;
                const bbData = data.filter(d => d.bbs && d.bbs[key]);
                if (bbData.length > 0) {
                    const up = chartRef.current.addSeries(LineSeries, baseOptions);
                    const lo = chartRef.current.addSeries(LineSeries, baseOptions);
                    up.setData(bbData.map(d => ({ time: d.time, value: d.bbs[key].up })));
                    lo.setData(bbData.map(d => ({ time: d.time, value: d.bbs[key].dn })));
                    indicatorSeriesRef.current.push(up, lo);
                }
            } else if (conf.type === 'ichimoku') {
                const key = `ICHI${conf.period.replace(/,/g, '_')}`;
                const ichiData = data.filter(d => d.ichis && d.ichis[key]);
                if (ichiData.length > 0) {
                    const sa = chartRef.current.addSeries(LineSeries, baseOptions);
                    const sb = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: conf.color + '80' });
                    sa.setData(ichiData.map(d => ({ time: d.time, value: d.ichis[key].sa })));
                    sb.setData(ichiData.map(d => ({ time: d.time, value: d.ichis[key].sb })));
                    indicatorSeriesRef.current.push(sa, sb);
                }
            } else if (['rsi', 'mfi', 'cci'].includes(conf.type)) {
                const key = `${conf.type.toUpperCase()}${conf.period}`;
                const paneId = `pane_${key}`;
                const oscData = data.filter(d => d.oscillators && d.oscillators[key]).map(d => ({ time: d.time, value: d.oscillators[key] }));
                if (oscData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(oscData);
                    chartRef.current.priceScale(paneId).applyOptions({ scaleMargins: { top: 0.8, bottom: 0.05 } });
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'volume') {
                const volData = data.filter(d => d.volume).map(d => ({
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? 'rgba(226, 16, 16, 0.6)' : 'rgba(0, 81, 255, 0.6)'
                }));
                if (volData.length > 0) {
                    const s = chartRef.current.addSeries(HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
                    s.setData(volData);
                    // volume 시리즈 추가 후 price scale 설정
                    chartRef.current.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'vol_sma' || conf.type === 'volma') {
                const key = `VOLMA${conf.period}`;
                const volmaData = data.filter(d => d.volumes && d.volumes[key]).map(d => ({ time: d.time, value: d.volumes[key] }));
                if (volmaData.length > 0) {
                    // vol price scale이 없으면 먼저 생성 (volume 시리즈가 없는 경우)
                    try {
                        chartRef.current.priceScale('vol');
                    } catch (e) {
                        // vol price scale이 없으면 더미 시리즈로 생성
                        const dummyVol = chartRef.current.addSeries(HistogramSeries, { priceFormat: { type: 'volume' }, priceScaleId: 'vol', visible: false });
                        dummyVol.setData([]);
                        chartRef.current.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
                    }
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: 'vol' });
                    s.setData(volmaData);
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'atr') {
                const key = `ATR${conf.period}`;
                const paneId = `pane_${key}`;
                const atrData = data.filter(d => d.atrs && d.atrs[key]).map(d => ({ time: d.time, value: d.atrs[key] }));
                if (atrData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(atrData);
                    chartRef.current.priceScale(paneId).applyOptions({ scaleMargins: { top: 0.8, bottom: 0.05 } });
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'psar') {
                const psarData = data.filter(d => d.psars && d.psars['PSAR']).map(d => ({ time: d.time, value: d.psars['PSAR'] }));
                if (psarData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, { 
                        ...baseOptions, 
                        lineStyle: 2, 
                        lineWidth: 0, 
                        markerType: 'circle',
                        markerSize: 3
                    });
                    s.setData(psarData);
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'adx') {
                const key = `ADX${conf.period}`;
                const paneId = `pane_${key}`;
                const adxData = data.filter(d => d.adxs && d.adxs[key]);
                if (adxData.length > 0) {
                    const sAdx = chartRef.current.addSeries(LineSeries, { ...baseOptions, priceScaleId: paneId });
                    const sPlus = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: '#4CAF50', lineWidth: 1, priceScaleId: paneId });
                    const sMinus = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: '#F44336', lineWidth: 1, priceScaleId: paneId });
                    sAdx.setData(adxData.map(d => ({ time: d.time, value: d.adxs[key].adx })));
                    sPlus.setData(adxData.map(d => ({ time: d.time, value: d.adxs[key].plus })));
                    sMinus.setData(adxData.map(d => ({ time: d.time, value: d.adxs[key].minus })));
                    chartRef.current.priceScale(paneId).applyOptions({ scaleMargins: { top: 0.75, bottom: 0.05 } });
                    indicatorSeriesRef.current.push(sAdx, sPlus, sMinus);
                }
            } else if (conf.type === 'donchian') {
                const key = `DC${conf.period}`;
                const dcData = data.filter(d => d.donchians && d.donchians[key]);
                if (dcData.length > 0) {
                    const up = chartRef.current.addSeries(LineSeries, baseOptions);
                    const lo = chartRef.current.addSeries(LineSeries, baseOptions);
                    up.setData(dcData.map(d => ({ time: d.time, value: d.donchians[key].up })));
                    lo.setData(dcData.map(d => ({ time: d.time, value: d.donchians[key].lo })));
                    indicatorSeriesRef.current.push(up, lo);
                }
            }
        });
    }, [data, configs]);

    return <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />;
};

export default ChartComponent;