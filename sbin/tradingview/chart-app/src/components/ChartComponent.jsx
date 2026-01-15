import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries } from 'lightweight-charts';

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
                const lineData = data.filter(d => d.mas[key]).map(d => ({ time: d.time, value: d.mas[key] }));
                if (lineData.length > 0) {
                    const s = chartRef.current.addSeries(LineSeries, baseOptions);
                    s.setData(lineData);
                    indicatorSeriesRef.current.push(s);
                }
            } else if (conf.type === 'bollinger') {
                const key = `BB${conf.period}`;
                const bbData = data.filter(d => d.bbs[key]);
                if (bbData.length > 0) {
                    const up = chartRef.current.addSeries(LineSeries, baseOptions);
                    const lo = chartRef.current.addSeries(LineSeries, baseOptions);
                    up.setData(bbData.map(d => ({ time: d.time, value: d.bbs[key].up })));
                    lo.setData(bbData.map(d => ({ time: d.time, value: d.bbs[key].dn })));
                    indicatorSeriesRef.current.push(up, lo);
                }
            } else if (conf.type === 'ichimoku') {
                const key = `ICHI${conf.period.replace(/,/g, '_')}`;
                const ichiData = data.filter(d => d.ichis[key]);
                if (ichiData.length > 0) {
                    const sa = chartRef.current.addSeries(LineSeries, baseOptions);
                    const sb = chartRef.current.addSeries(LineSeries, { ...baseOptions, color: conf.color + '80' });
                    sa.setData(ichiData.map(d => ({ time: d.time, value: d.ichis[key].sa })));
                    sb.setData(ichiData.map(d => ({ time: d.time, value: d.ichis[key].sb })));
                    indicatorSeriesRef.current.push(sa, sb);
                }
            }
        });
    }, [data, configs]);

    return <div ref={chartContainerRef} style={{ width: '100%', height: '100%' }} />;
};

export default ChartComponent;