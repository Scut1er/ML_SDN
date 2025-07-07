# ML Balancing Frontend Integration - Изменения

## Что добавлено

Переключатель ML балансировки интегрирован в существующую систему настроек проекта с минимальными изменениями.

## Изменения в файлах

### 1. **Frontend Types** (SDN-sim-mod-frontend/src/context/BoardSettingsContext/meta.ts)

```typescript
export interface ISendableBoardSettingsConfig {
    modelsSettings: ISendableBoardSettingsConfigBlock;
    qualityOfServiceSettings: ISendableBoardSettingsConfigBlock;
    mlBalancingSettings: ISendableBoardSettingsConfigBlock; // ← ДОБАВЛЕНО
}
```

### 2. **Backend Types** (SDN-sim-mod-backend/src/domains/Board/meta.ts)

```typescript
// Добавлен интерфейс
export interface ISendableBoardSettingsConfig {
    modelsSettings: ISendableBoardSettingsConfigBlock;
    qualityOfServiceSettings: ISendableBoardSettingsConfigBlock;
    mlBalancingSettings: ISendableBoardSettingsConfigBlock; // ← ДОБАВЛЕНО
}

// Добавлен заголовок
export const enum BoardSettingsConfigBlocksTitles {
    BOARD_SETTINGS = "Board settings",
    QUALITY_OF_SERVICE_ACTIVE = "QOS settings",
    ML_BALANCING_ACTIVE = "ML Balancing", // ← ДОБАВЛЕНО
}
```

### 3. **Constants** (SDN-sim-mod-backend/src/utils/constants/index.ts)

```typescript
export const DEFAULT_IS_ML_BALANCING_ACTIVE: boolean = false; // ← ДОБАВЛЕНО
```

### 4. **Settings Config Service** (SDN-sim-mod-backend/src/services/BoardSettingsConfigService/index.ts)

```typescript
// Добавлен импорт
import { DEFAULT_IS_ML_BALANCING_ACTIVE, ... } from "../../utils/constants";

// Добавлен блок настроек
public static getDefaultBoardSettingsConfig(): ISendableBoardSettingsConfig {
    return {
        // ... существующие настройки
        mlBalancingSettings: { // ← ДОБАВЛЕНО
            title: BoardSettingsConfigBlocksTitles.ML_BALANCING_ACTIVE,
            isActive: DEFAULT_IS_ML_BALANCING_ACTIVE,
            fields: {},
        },
    };
}
```

### 5. **Board Logic** (SDN-sim-mod-backend/src/domains/Board/index.ts)

```typescript
public updateSettingsConfig(sendedSettingsConfig: ISendableBoardSettingsConfig): void {
    // ... существующие настройки
    this.settingsConfig.isMLBalancingActive = sendedSettingsConfig.mlBalancingSettings.isActive!; // ← ДОБАВЛЕНО
}
```

### 6. **ML Toggle Component** (НОВЫЙ файл)

- `SDN-sim-mod-frontend/src/components/MLBalancingToggle/index.tsx` 
- `SDN-sim-mod-frontend/src/components/MLBalancingToggle/style.css`

## Как работает

1. **Переключатель автоматически появляется** в настройках как третий блок "ML Balancing"
2. **Использует существующий BoardSettingsBlock** - никаких специальных интеграций не нужно
3. **Автоматически сохраняется** в настройках и передается на бэкенд
4. **Board автоматически создает MLBalancer** когда `isMLBalancingActive = true`

## Принцип интеграции

```
Frontend Settings → Backend Settings → Board Logic
       ↓                    ↓              ↓
mlBalancingSettings → isMLBalancingActive → MLBalancer vs Balancer
```

## Что НЕ нужно менять

- ❌ Компоненты настроек (BoardSettings, BoardSettingsBlock)
- ❌ Логика отправки настроек
- ❌ WebSocket коммуникация
- ❌ Логика создания балансировщиков

## Результат

Переключатель ML балансировки **автоматически появится** в интерфейсе настроек рядом с QoS переключателем. При включении система будет использовать MLBalancer вместо обычного Balancer.

Интеграция выполнена с **минимальными изменениями** в основной проект! 