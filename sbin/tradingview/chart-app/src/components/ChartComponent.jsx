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

        // 메인 가격 차트의 scaleMargins 설정 (하단에 여유 공간 확보)
        chart.priceScale('right').applyOptions({
            scaleMargins: { top: 0.1, bottom: 0.2 }
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
        
        // 모든 업데이트를 requestAnimationFrame으로 배치 처리하여 깜빡임 방지
        requestAnimationFrame(() => {
            // 우리가 가진 가장 과거 시간 저장
            firstTimeRef.current = data[0].time;
            
            // 캔들 데이터 세팅 (null/NaN 값 필터링)
            const validCandleData = data.filter(d => 
                d.open != null && !Number.isNaN(d.open) &&
                d.high != null && !Number.isNaN(d.high) &&
                d.low != null && !Number.isNaN(d.low) &&
                d.close != null && !Number.isNaN(d.close)
            );

            // 기존 보조지표를 Map으로 관리 (label -> series)하여 재사용
            const existingSeriesMap = new Map();
            indicatorSeriesRef.current.forEach(({ series, label }) => {
                existingSeriesMap.set(label, series);
            });
            
            // indicatorSeriesRef를 초기화하여 중복 방지
            indicatorSeriesRef.current = [];
            
            // 사용할 보조지표 label 집합을 추적
            const usedLabels = new Set();
            
            // 시리즈를 재사용하거나 새로 추가하는 헬퍼 함수
            const getOrCreateSeries = (label, seriesType, options) => {
                usedLabels.add(label);
                let series = existingSeriesMap.get(label);
                if (series) {
                    try {
                        // 기존 시리즈 재사용 - 옵션 업데이트만 수행
                        // priceScaleId는 시리즈 생성 시에만 설정 가능하므로 변경 불가
                        // 옵션에서 priceScaleId 제외하고 업데이트
                        const { priceScaleId, ...updateOptions } = options;
                        if (Object.keys(updateOptions).length > 0) {
                            series.applyOptions(updateOptions);
                        }
                        return series;
                    } catch (e) {
                        // 시리즈가 유효하지 않으면 재생성
                        try {
                            chartRef.current.removeSeries(series);
                        } catch (removeError) {
                            // 이미 제거되었을 수 있음
                        }
                        series = chartRef.current.addSeries(seriesType, options);
                        existingSeriesMap.set(label, series);
                        return series;
                    }
                } else {
                    // 새 시리즈 추가
                    series = chartRef.current.addSeries(seriesType, options);
                    existingSeriesMap.set(label, series);
                    return series;
                }
            };

        // 보조지표를 그룹별로 분류
        const independentIndicators = [];
        const priceIndicators = [];

        configs.forEach(conf => {
            if (['sma', 'ema', 'wma', 'bollinger', 'ichimoku', 'psar', 'donchian'].includes(conf.type)) {
                priceIndicators.push(conf);
            } else {
                // volume, volma, rsi, mfi, cci, atr, adx 등 모두 독립 보조지표로 처리
                independentIndicators.push(conf);
            }
        });

        // volume과 volma가 모두 있으면 하나로 카운트
        const hasVolume = independentIndicators.some(conf => conf.type === 'volume');
        const hasVolma = independentIndicators.some(conf => conf.type === 'vol_sma' || conf.type === 'volma');
        const volumeVolmaCount = (hasVolume && hasVolma) ? 1 : (hasVolume || hasVolma ? 1 : 0);
        const otherIndicatorsCount = independentIndicators.filter(conf => 
            conf.type !== 'volume' && conf.type !== 'vol_sma' && conf.type !== 'volma'
        ).length;
        
        // 동적 비율 계산: price는 항상 존재, 나머지 보조지표들이 추가될 때마다 비율 조정
        // volume과 volma는 같은 pane을 사용하므로 하나로 카운트
        const totalNonPriceIndicators = volumeVolmaCount + otherIndicatorsCount;
        
        // price 비율 계산 (나머지 보조지표 개수에 따라 조정)
        // 보조지표가 추가될 때 각 보조지표가 최소 크기를 유지하도록 price 비율을 더 빠르게 줄임
        let priceBottom = 0.1;
        if (totalNonPriceIndicators === 0) {
            priceBottom = 0.1; // price만 있을 때
        } else if (totalNonPriceIndicators === 1) {
            priceBottom = 0.3; // 보조지표 1개: price 70%, 보조지표 30%
        } else if (totalNonPriceIndicators === 2) {
            priceBottom = 0.4; // 보조지표 2개: price 60%, 각 보조지표 약 20%
        } else if (totalNonPriceIndicators === 3) {
            priceBottom = 0.45; // 보조지표 3개: price 55%, 각 보조지표 약 15%
        } else if (totalNonPriceIndicators === 4) {
            priceBottom = 0.5; // 보조지표 4개: price 50%, 각 보조지표 약 12.5%
        } else if (totalNonPriceIndicators === 5) {
            priceBottom = 0.55; // 보조지표 5개: price 45%, 각 보조지표 약 11%
        } else {
            priceBottom = 0.6; // 보조지표 6개 이상: price 40%, 각 보조지표 비율 분배
        }

        // price scaleMargins 업데이트 (캔들 차트는 항상 있음)
        chartRef.current.priceScale('right').applyOptions({
            scaleMargins: { top: 0.1, bottom: priceBottom }
        });

        
        const calculateMargins = (index, total) => {
            // 지표가 없을 경우 기본값
            if (total === 0) return { top: 0.95, bottom: 0.05 };
        
            // index가 범위를 벗어나지 않도록 보정
            const safeIndex = Math.max(0, Math.min(index, total - 1));
        
            // priceBottom이 전체 가용 공간의 크기 (예: 0.3)
            const availableSpace = priceBottom; 
            
            // 하나의 보조지표가 차지할 높이 (예: 0.3 / 2 = 0.15)
            const indicatorHeight = availableSpace / total;
        
            // 상단(index 0)부터 순차 배치 계산
            // index 0: top = 1 - 0.3 = 0.7,  bottom = 0.15 * (2 - 1 - 0) = 0.15
            // index 1: top = 1 - 0.3 + 0.15 = 0.85, bottom = 0.15 * (2 - 1 - 1) = 0.0
            
            const top = (1 - availableSpace) + (safeIndex * indicatorHeight);
            const bottom = indicatorHeight * (total - 1 - safeIndex);
        
            // bottom이 음수가 되지 않도록 보정
            return { 
                top: Math.max(0, Math.min(1, top)), 
                bottom: Math.max(0, Math.min(1, bottom)) 
            };
        };

        // 1. Price 그룹 지표들 (MA, BB, ICHI, PSAR, DC) - priceScaleId 없이 기본 'right' 사용
        priceIndicators.forEach(conf => {
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
                    const s = getOrCreateSeries(key, LineSeries, baseOptions);
                    s.setData(lineData);
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'bollinger') {
                const key = `BB${conf.period}`;
                const bbData = data.filter(d => d.bbs && d.bbs[key]);
                if (bbData.length > 0) {
                    const up = getOrCreateSeries(`${key}_UP`, LineSeries, baseOptions);
                    const lo = getOrCreateSeries(`${key}_DN`, LineSeries, baseOptions);
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
                    const sa = getOrCreateSeries(`${key}_SA`, LineSeries, baseOptions);
                    const sb = getOrCreateSeries(`${key}_SB`, LineSeries, { ...baseOptions, color: conf.color + '80' });
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
            } else if (conf.type === 'psar') {
                const psarData = data
                    .filter(d => d.psars && d.psars['PSAR'] != null && !Number.isNaN(d.psars['PSAR']))
                    .map(d => ({ time: d.time, value: d.psars['PSAR'] }));
                if (psarData.length > 0) {
                    const s = getOrCreateSeries('PSAR', LineSeries, { 
                        ...baseOptions, 
                        lineStyle: 2, 
                        lineWidth: 0, 
                        markerType: 'circle',
                        markerSize: 3
                    });
                    s.setData(psarData);
                    indicatorSeriesRef.current.push({ series: s, label: 'PSAR' });
                }
            } else if (conf.type === 'donchian') {
                const key = `DC${conf.period}`;
                const dcData = data.filter(d => d.donchians && d.donchians[key]);
                if (dcData.length > 0) {
                    const up = getOrCreateSeries(`${key}_UP`, LineSeries, baseOptions);
                    const lo = getOrCreateSeries(`${key}_LO`, LineSeries, baseOptions);
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

        // volume과 volma가 같은 pane을 사용하도록 volumePaneId 추적
        let volumePaneId = 'pane_VOL';

        // 2. 독립 보조지표들 (RSI, MFI, CCI, ATR, ADX, VOL, VOLMA) - 각각 고유한 priceScaleId 사용 (상단부터 순차 배치)
        // 실제 pane 인덱스를 추적 (volume/volma는 하나로 카운트)
        let actualPaneIndex = 0;
        let volumePaneAssigned = false;
        let volumePaneIndex = null; // volume이 할당된 실제 pane 인덱스를 저장
        
        independentIndicators.forEach((conf, idx) => {
            const baseOptions = { 
                color: conf.color, 
                lineWidth: conf.width || 2, 
                priceLineVisible: false, 
                lastValueVisible: false 
            };
            
            // paneIndex 계산: volume과 volma는 같은 pane을 사용
            let paneIndex;
            if (conf.type === 'volume' || conf.type === 'vol_sma' || conf.type === 'volma') {
                if (!volumePaneAssigned) {
                    // volume/volma의 첫 번째 항목이 실제 pane 인덱스 결정
                    paneIndex = actualPaneIndex;
                    volumePaneIndex = actualPaneIndex; // volume pane 인덱스 저장
                    volumePaneAssigned = true;
                    actualPaneIndex++; // volume/volma pane 할당 후 다음 pane으로
                } else {
                    // volma는 volume과 같은 pane 사용
                    paneIndex = volumePaneIndex; // 저장된 volume pane 인덱스 사용
                }
            } else {
                // volume/volma가 아닌 지표는 순차적으로 pane 인덱스 할당
                paneIndex = actualPaneIndex;
                actualPaneIndex++;
            }
            
            // paneIndex가 범위를 벗어나지 않도록 보정
            if (paneIndex >= totalNonPriceIndicators) {
                paneIndex = Math.max(0, totalNonPriceIndicators - 1);
            }
            
            // bottom margin이 음수가 되지 않도록 보정
            const margins = calculateMargins(paneIndex, totalNonPriceIndicators);
            if (margins.bottom < 0) {
                margins.bottom = 0;
            }
            if (margins.top < 0) {
                margins.top = 0;
            }
            if (margins.top > 1) {
                margins.top = 1;
            }
            if (margins.bottom > 1) {
                margins.bottom = 1;
            }
            
            if (['rsi', 'mfi', 'cci'].includes(conf.type)) {
                const key = `${conf.type.toUpperCase()}${conf.period}`;
                const paneId = `pane_${key}`;
                const oscData = data
                    .filter(d => d.oscillators && d.oscillators[key] != null && !Number.isNaN(d.oscillators[key]))
                    .map(d => ({ time: d.time, value: d.oscillators[key] }));
                if (oscData.length > 0) {
                    const s = getOrCreateSeries(key, LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(oscData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: margins,
                        position: 'right'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'atr') {
                const key = `ATR${conf.period}`;
                const paneId = `pane_${key}`;
                const atrData = data
                    .filter(d => d.atrs && d.atrs[key] != null && !Number.isNaN(d.atrs[key]))
                    .map(d => ({ time: d.time, value: d.atrs[key] }));
                if (atrData.length > 0) {
                    const s = getOrCreateSeries(key, LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(atrData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: margins,
                        position: 'right'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            } else if (conf.type === 'adx') {
                const key = `ADX${conf.period}`;
                const paneId = `pane_${key}`;
                const adxData = data.filter(d => d.adxs && d.adxs[key]);
                if (adxData.length > 0) {
                    const sAdx = getOrCreateSeries(key, LineSeries, { ...baseOptions, priceScaleId: paneId });
                    const sPlus = getOrCreateSeries(`${key}_PLUS`, LineSeries, { ...baseOptions, color: '#4CAF50', lineWidth: 1, priceScaleId: paneId });
                    const sMinus = getOrCreateSeries(`${key}_MINUS`, LineSeries, { ...baseOptions, color: '#F44336', lineWidth: 1, priceScaleId: paneId });
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
                        scaleMargins: margins,
                        position: 'right'
                    });
                    indicatorSeriesRef.current.push(
                        { series: sAdx, label: key },
                        { series: sPlus, label: `${key}_PLUS` },
                        { series: sMinus, label: `${key}_MINUS` },
                    );
                }
            } else if (conf.type === 'volume') {
                const paneId = volumePaneId;
                const volData = data.filter(d => d.volume != null).map(d => ({
                    time: d.time,
                    value: d.volume,
                    color: d.close >= d.open ? 'rgba(226, 16, 16, 0.6)' : 'rgba(0, 81, 255, 0.6)'
                }));
                if (volData.length > 0) {
                    const s = getOrCreateSeries('VOL', HistogramSeries, { 
                        priceFormat: { type: 'volume' }, 
                        priceScaleId: paneId,
                        priceLineVisible: false,
                        lastValueVisible: false
                    });
                    s.setData(volData);
                    chartRef.current.priceScale(paneId).applyOptions({ 
                        scaleMargins: margins,
                        position: 'right'
                    });
                    indicatorSeriesRef.current.push({ series: s, label: 'VOL' });
                }
            } else if (conf.type === 'vol_sma' || conf.type === 'volma') {
                const key = `VOLMA${conf.period}`;
                const paneId = volumePaneId; // volume과 같은 pane 사용
                const volmaData = data
                    .filter(d => d.volumes && d.volumes[key] != null && !Number.isNaN(d.volumes[key]))
                    .map(d => ({ time: d.time, value: d.volumes[key] }));
                if (volmaData.length > 0) {
                    const s = getOrCreateSeries(key, LineSeries, { ...baseOptions, priceScaleId: paneId });
                    s.setData(volmaData);
                    // volume이 이미 priceScale을 설정했을 수 있으므로, 없을 때만 설정
                    try {
                        chartRef.current.priceScale(paneId).applyOptions({ 
                            scaleMargins: margins,
                            position: 'right'
                        });
                    } catch (e) {
                        // 이미 설정되어 있으면 무시
                    }
                    indicatorSeriesRef.current.push({ series: s, label: key });
                }
            }
        });

            // 사용되지 않은 시리즈 제거 (configs에서 제거된 보조지표)
            // 모든 시리즈를 한 번에 수집한 후 한 번에 제거하여 렌더링 최소화
            const seriesToRemove = [];
            existingSeriesMap.forEach((series, label) => {
                if (!usedLabels.has(label)) {
                    seriesToRemove.push({ series, label });
                }
            });
            
            // 모든 시리즈를 한 번에 제거
            seriesToRemove.forEach(({ series, label }) => {
                try {
                    // 시리즈가 유효한지 확인 후 제거
                    if (series && chartRef.current) {
                        chartRef.current.removeSeries(series);
                    }
                } catch (e) {
                    // 이미 제거되었거나 유효하지 않은 시리즈는 무시
                }
                existingSeriesMap.delete(label);
            });

            // 모든 보조지표 추가가 완료된 후 마지막에 캔들 데이터 설정
            // 이렇게 하면 차트가 한 번만 렌더링되어 깜빡임이 최소화됨
            candleSeriesRef.current.setData(validCandleData);
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