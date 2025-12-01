# 🔄 LIFECYCLE STATE FLOW - API Control

## 📊 TÓM TẮT

Khi gọi API `/api/alarm/control`, hệ thống **TỰ ĐỘNG CẬP NHẬT** `lifecycle_state`:

| API Call        | Alarm Action                  | Lifecycle State Transition      |
| --------------- | ----------------------------- | ------------------------------- |
| `enabled=true`  | ✅ **BẬT** alarm ngay lập tức | `ANY_STATE` → `ALARM_ACTIVATED` |
| `enabled=false` | ❌ **TẮT** alarm ngay lập tức | `ANY_STATE` → `RESOLVED`        |

---

## 🎯 LOGIC THIẾT KẾ

### ✅ Khi `enabled=true` (BẬT ALARM)

**Ý nghĩa:** User muốn BẬT alarm ngay → Coi như đang escalate event thành alarm

**Hành động:**

1. 🔊 Phát alarm âm thanh ngay lập tức
2. 📝 Update database:
   ```sql
   UPDATE event_detections
   SET
       lifecycle_state = 'ALARM_ACTIVATED',  -- ← Chuyển sang ALARM_ACTIVATED
       escalated_at = NOW(),                  -- ← Ghi thời điểm escalate
       last_action_at = NOW(),
       notes = '... Alarm ACTIVATED via API by [user]'
   WHERE event_id = ?
   ```

**Kết quả:**

- ✅ Event chuyển sang trạng thái `ALARM_ACTIVATED`
- ✅ Có timestamp `escalated_at` để track
- ✅ Log rõ "ACTIVATED via API" trong notes
- ✅ Tương đương với auto-alarm của Worker

---

### ❌ Khi `enabled=false` (TẮT ALARM)

**Ý nghĩa:** Alarm đã phát, đã có tác động → Coi như đã xử lý xong

**Hành động:**

1. 🔇 Dừng alarm âm thanh ngay lập tức
2. 📝 Update database:
   ```sql
   UPDATE event_detections
   SET
       lifecycle_state = 'RESOLVED',  -- ← Chuyển sang RESOLVED
       resolved_at = NOW(),            -- ← Ghi thời điểm resolve
       last_action_at = NOW(),
       notes = '... Alarm RESOLVED via API by [user]'
   WHERE event_id = ?
   ```

**Kết quả:**

- ✅ Event chuyển sang trạng thái `RESOLVED` (đã xử lý xong)
- ✅ Có timestamp `resolved_at` để track
- ✅ Log rõ "RESOLVED via API" trong notes
- ✅ Event không còn trong danh sách active alarms

---

## 🔄 FULL LIFECYCLE FLOW

### Flow 1: Auto-Alarm (Worker)

```
1. Event Created (danger/warning)
   ↓
   lifecycle_state = 'NOTIFIED'

2. [Đợi 30 giây...]
   ↓
   EventLifecycleWorker check

3. Chưa xử lý sau 30s
   ↓
   Worker trigger alarm
   ↓
   lifecycle_state = 'ALARM_ACTIVATED'
   escalated_at = NOW()

4. [Đợi 30 giây với status = normal...]
   ↓
   Worker auto-resolve
   ↓
   lifecycle_state = 'RESOLVED'
   resolved_at = NOW()
```

### Flow 2: Manual Control (API)

```
1. Event Created (bất kỳ state nào)
   ↓
   lifecycle_state = 'NOTIFIED' / 'ACKNOWLEDGED' / ...

2. User gọi API enabled=true
   ↓
   API trigger alarm ngay
   ↓
   lifecycle_state = 'ALARM_ACTIVATED'  ← MANUAL ESCALATE
   escalated_at = NOW()

3. User gọi API enabled=false
   ↓
   API stop alarm ngay
   ↓
   lifecycle_state = 'RESOLVED'  ← MANUAL RESOLVE
   resolved_at = NOW()
```

### Flow 3: Mix (Worker + API)

```
1. Event Created
   ↓
   lifecycle_state = 'NOTIFIED'

2. Worker auto-alarm sau 30s
   ↓
   lifecycle_state = 'ALARM_ACTIVATED'

3. User gọi API enabled=false
   ↓
   API stop alarm + resolve
   ↓
   lifecycle_state = 'RESOLVED'  ← Override worker

✅ User intervention có quyền cao nhất!
```

---

## 📋 STATE TRANSITION TABLE

| Current State     | API `enabled=true`  | API `enabled=false` | Worker Auto-Alarm   | Worker Auto-Resolve |
| ----------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| `NOTIFIED`        | → `ALARM_ACTIVATED` | → `RESOLVED`        | → `ALARM_ACTIVATED` | ❌ (chưa có alarm)  |
| `ACKNOWLEDGED`    | → `ALARM_ACTIVATED` | → `RESOLVED`        | ❌ (đã ack)         | ❌ (không có alarm) |
| `ALARM_ACTIVATED` | ⚠️ Đã active rồi    | → `RESOLVED`        | ❌ (đã active)      | → `RESOLVED`        |
| `RESOLVED`        | → `ALARM_ACTIVATED` | ⚠️ Đã resolve rồi   | ❌ (đã resolve)     | ❌ (đã resolve)     |
| `CANCELLED`       | → `ALARM_ACTIVATED` | → `RESOLVED`        | ❌ (đã cancel)      | ❌ (đã cancel)      |

**Chú thích:**

- ✅ = Transition thành công
- ❌ = Worker không xử lý state này
- ⚠️ = API vẫn chạy được nhưng state không đổi (idempotent)

---

## 🎯 TẠI SAO THIẾT KẾ NHƯ VẬY?

### ✅ Ưu điểm

1. **Đơn giản và trực quan:**

   - `enabled=true` = "Tôi muốn BẬT alarm" → State phải là ALARM_ACTIVATED
   - `enabled=false` = "Tôi muốn TẮT alarm" → Alarm đã phát rồi, coi như xử lý xong → RESOLVED

2. **Consistency với auto-alarm:**

   - Worker: auto-alarm → ALARM_ACTIVATED
   - API: manual alarm → ALARM_ACTIVATED
   - Cùng 1 state cho cùng 1 hành động!

3. **Audit trail rõ ràng:**

   ```
   lifecycle_state = ALARM_ACTIVATED
   + escalated_at = 2025-11-30 10:30:00
   + notes = "Alarm ACTIVATED via API by admin"

   → Biết rõ: alarm được trigger lúc nào, bởi ai, qua kênh nào
   ```

4. **Integration với mobile app:**

   - Mobile app query: `WHERE lifecycle_state = 'ALARM_ACTIVATED'`
   - Mobile app hiển thị danh sách events đang alarm
   - API stop → State RESOLVED → Events biến mất khỏi danh sách active

5. **Không conflict với Worker:**
   - Worker check: `lifecycle_state = 'NOTIFIED'` mới auto-alarm
   - Nếu API đã ALARM_ACTIVATED rồi → Worker skip (không trigger lại)
   - Nếu API đã RESOLVED rồi → Worker skip (không cần xử lý)

### ⚠️ Lưu Ý

1. **API có quyền override Worker:**

   ```
   Worker auto-alarm (30s) → ALARM_ACTIVATED
   ↓
   User gọi API enabled=false → RESOLVED
   ↓
   Worker sẽ KHÔNG auto-resolve lại (vì state đã RESOLVED)
   ```

2. **Có thể trigger lại alarm cho event đã RESOLVED:**

   ```sql
   -- Event đã RESOLVED
   lifecycle_state = 'RESOLVED'

   -- User gọi API enabled=true
   → lifecycle_state = 'ALARM_ACTIVATED'

   ✅ Cho phép re-escalate nếu cần!
   ```

3. **Notes field ghi rõ lịch sử:**
   ```
   [2025-11-30 10:00:00] Event created
   [2025-11-30 10:00:35] Auto-alarm activated after 30s timeout
   [2025-11-30 10:02:00] Alarm RESOLVED via API by admin - User confirmed false alarm
   [2025-11-30 10:05:00] Alarm ACTIVATED via API by supervisor - Escalate again for review
   ```

---

## 📊 COMPARISON: OLD vs NEW

### ❌ OLD Design (Không khuyến khích)

```python
# API chỉ trigger alarm, KHÔNG đổi state
enabled=true  → Chỉ phát alarm
enabled=false → Chỉ tắt alarm

# Problem:
lifecycle_state = 'NOTIFIED'  # Vẫn NOTIFIED dù alarm đã phát!
→ Worker sẽ trigger lại sau 30s (duplicate alarm)
→ Mobile app không biết alarm đang phát (state vẫn NOTIFIED)
→ Khó tracking: alarm nào đang active?
```

### ✅ NEW Design (Hiện tại)

```python
# API trigger alarm + update state
enabled=true  → Phát alarm + ALARM_ACTIVATED
enabled=false → Tắt alarm + RESOLVED

# Benefits:
lifecycle_state = 'ALARM_ACTIVATED'  # State phản ánh đúng thực tế!
→ Worker skip (không trigger lại)
→ Mobile app query được danh sách active alarms
→ Dễ tracking: escalated_at, resolved_at timestamps
→ Audit trail đầy đủ trong notes
```

---

## 🧪 TESTING SCENARIOS

### Test 1: Manual Trigger → Manual Resolve

```bash
# 1. Tạo event mới (status=danger)
Event created → lifecycle_state = 'NOTIFIED'

# 2. Gọi API BẬT alarm
curl -X POST http://localhost:8000/api/alarm/control \
  -d '{"event_id":"...", "enabled":true}'

→ Alarm phát
→ lifecycle_state = 'ALARM_ACTIVATED'
→ escalated_at = NOW()

# 3. Gọi API TẮT alarm
curl -X POST http://localhost:8000/api/alarm/control \
  -d '{"event_id":"...", "enabled":false}'

→ Alarm dừng
→ lifecycle_state = 'RESOLVED'
→ resolved_at = NOW()
```

### Test 2: Worker Auto → API Override

```bash
# 1. Event created
lifecycle_state = 'NOTIFIED'

# 2. Đợi 30s → Worker auto-alarm
lifecycle_state = 'ALARM_ACTIVATED'

# 3. User gọi API TẮT (không đợi worker auto-resolve)
curl -X POST http://localhost:8000/api/alarm/control \
  -d '{"enabled":false}'

→ lifecycle_state = 'RESOLVED'
→ notes = "Alarm RESOLVED via API by user"
✅ User can override automation!
```

### Test 3: Re-escalate Event

```bash
# 1. Event đã RESOLVED
lifecycle_state = 'RESOLVED'

# 2. Admin quyết định escalate lại
curl -X POST http://localhost:8000/api/alarm/control \
  -d '{"enabled":true}'

→ Alarm phát lại
→ lifecycle_state = 'ALARM_ACTIVATED'
→ notes = "Alarm ACTIVATED via API by admin - Re-escalated for review"
✅ Flexible re-escalation!
```

---

## 💡 BEST PRACTICES

### 1. Query Active Alarms (Mobile App)

```sql
-- Lấy danh sách alarms đang active
SELECT
    event_id,
    event_type,
    status,
    escalated_at,
    notes
FROM event_detections
WHERE lifecycle_state = 'ALARM_ACTIVATED'
  AND is_canceled = FALSE
ORDER BY escalated_at DESC;
```

### 2. Tracking Manual vs Auto

```sql
-- Events được manual trigger
SELECT * FROM event_detections
WHERE notes LIKE '%ACTIVATED via API%';

-- Events được auto-alarm bởi Worker
SELECT * FROM event_detections
WHERE notes LIKE '%Auto-alarm activated after 30s%';
```

### 3. Performance Query

```sql
-- Index cho performance
CREATE INDEX idx_lifecycle_escalated
ON event_detections(lifecycle_state, escalated_at DESC)
WHERE lifecycle_state = 'ALARM_ACTIVATED';
```

---

## 📝 SUMMARY

**API Control Logic:**

1. ✅ **`enabled=true`**

   - Phát alarm ngay
   - Update `lifecycle_state = ALARM_ACTIVATED`
   - Set `escalated_at = NOW()`
   - Log "ACTIVATED via API"

2. ❌ **`enabled=false`**

   - Dừng alarm ngay
   - Update `lifecycle_state = RESOLVED`
   - Set `resolved_at = NOW()`
   - Log "RESOLVED via API"

3. 🤖 **Worker vẫn hoạt động bình thường:**

   - Auto-alarm: NOTIFIED → ALARM_ACTIVATED (30s)
   - Auto-resolve: ALARM_ACTIVATED → RESOLVED (30s normal)
   - Không conflict với API (check state trước khi action)

4. 🎯 **User có quyền override automation:**
   - Manual trigger override worker timing
   - Manual resolve override worker auto-resolve
   - Flexible re-escalation cho resolved events

**Kết luận:** Thiết kế vừa đơn giản, vừa powerful, vừa consistent! 🎉
