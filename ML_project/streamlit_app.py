"""
AI-based Personalized Learning Habit Analysis System  v2.0
===========================================================
Modules:
  1. Prediction   — smart feature engineering + confidence score
  2. Learning Assistant — goal tracking, time optimizer, study plan
  3. AI Coach     — weakness radar, bad-habit detection, coaching tips

Model fix: assignment_score & midterm_score are now estimated from
           study habits instead of using a fixed mean → realistic
           predictions (Poor ≈ 54 / Average ≈ 72 / Perfect ≈ 86)
"""

import io, os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH    = os.path.join(BASE_DIR,
               "Student Performance Analytics Dataset.csv")

FEATURE_COLS = [
    "gender", "study_hours_per_day", "attendance_percentage",
    "assignment_score", "midterm_score", "participation_score",
    "internet_access", "extra_classes", "parent_education", "sleep_hours",
]

# Fixed-value defaults for features the user does NOT control
HIDDEN_DEFAULTS = {
    "gender": 0, "internet_access": 1,
    "extra_classes": 0, "parent_education": 2, "sleep_hours": 7.0,
}

DATASET_MEAN  = 63.83   # from 10,000 student records
DATASET_TOP25 = 82.0    # 75th percentile

LEVELS = [
    (0,  40,  "🌱 Beginner",     "#e74c3c"),
    (40, 55,  "📚 Developing",   "#e67e22"),
    (55, 70,  "⚡ Intermediate", "#f1c40f"),
    (70, 85,  "🎯 Advanced",     "#2ecc71"),
    (85, 101, "🏆 Master",       "#9b59b6"),
]

# Matplotlib dark-theme palette
BG, PANEL, GRID = "#0e1117", "#1e2130", "#30363d"
WHITE = "white"


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown("""
    <style>
    .score-hero {
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        border: 1px solid #30363d; border-radius: 16px;
        padding: 28px 16px; text-align: center; margin-bottom: 12px;
    }
    .score-num  { font-size: 82px; font-weight: 900; line-height: 1.05; }
    .score-sub  { font-size: 20px; font-weight: 600; margin-top: 6px; opacity: .9; }

    .card {
        background: #1e2130; border: 1px solid #30363d;
        border-radius: 12px; padding: 16px 18px; margin: 6px 0;
    }
    .card-red    { border-left: 4px solid #e74c3c; }
    .card-orange { border-left: 4px solid #e67e22; }
    .card-green  { border-left: 4px solid #2ecc71; }
    .card-blue   { border-left: 4px solid #3498db; }
    .card-purple { border-left: 4px solid #9b59b6; }

    .feat-box {
        background: #1e2130; border: 1px solid #30363d;
        border-radius: 14px; padding: 22px; text-align: center;
    }
    .coach-tip {
        background: #162032; border: 1px solid #1e4a70;
        border-radius: 10px; padding: 14px 18px; margin: 6px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  ── cached per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Return fitted Pipeline from disk; auto-trains if missing."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return _train_model()


def _train_model():
    from sklearn.ensemble import (GradientBoostingRegressor,
                                   RandomForestRegressor, StackingRegressor)
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    with st.spinner("⏳ Đang huấn luyện mô hình lần đầu (~30s)…"):
        df = pd.read_csv(DATA_PATH)
        X_tr, _, y_tr, _ = train_test_split(
            df[FEATURE_COLS], df["overall_score"], test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("rf", RandomForestRegressor(
                        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                    ("gb", GradientBoostingRegressor(
                        n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)),
                ],
                final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)),
        ])
        pipe.fit(X_tr, y_tr)
    joblib.dump(pipe, MODEL_PATH)
    st.success("✅ Mô hình đã huấn luyện xong!")
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  ── the key model fix
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_hidden(study_hours: float, att_pct: float, focus: int) -> dict:
    """
    Estimate assignment_score & midterm_score from observable habits.
    Previously these were fixed at 65/63 causing a ceiling at ~70.

    Formula derived so that:
      Perfect (10h, 100%, focus=10) → assignment≈100, midterm≈100 → score≈86
      Average (5h,  92%, focus=7)  → assignment≈73,  midterm≈72  → score≈72
      Poor    (2h,  62%, focus=3)  → assignment≈50,  midterm≈45  → score≈54
    """
    assignment = np.clip(
        30 + (study_hours / 10) * 40 + (focus / 10) * 20 + (att_pct / 100) * 10,
        30.0, 99.9,
    )
    midterm = np.clip(
        20 + (study_hours / 10) * 35 + (focus / 10) * 30 + (att_pct / 100) * 15,
        25.0, 100.0,
    )
    return {"assignment_score": assignment, "midterm_score": midterm}


def build_row(study_hours: float, absences: int, focus: int) -> pd.DataFrame:
    """Build the full 10-feature vector for a given set of user inputs."""
    att_pct = max(40.0, (40 - absences) / 40 * 100)
    row = {
        **HIDDEN_DEFAULTS,
        "study_hours_per_day"   : float(study_hours),
        "attendance_percentage" : att_pct,
        "participation_score"   : focus * 10.0,
        **_estimate_hidden(float(study_hours), att_pct, focus),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def derived_metrics(study_hours: float, absences: int, focus: int) -> dict:
    """Extra engineered features shown in the UI."""
    att   = max(0.0, (40 - absences) / 40)
    eff   = (focus / 10) * min(study_hours / 8, 1.0)
    inten = study_hours * (focus / 10)
    return {
        "attendance_rate"  : round(att   * 100, 1),
        "study_efficiency" : round(eff   * 100, 1),
        "study_intensity"  : round(inten, 2),
    }


def predict(model, study_hours: float, absences: int, focus: int) -> float:
    """Predict score and clamp to [0, 100]."""
    return float(np.clip(
        model.predict(build_row(study_hours, absences, focus))[0], 0, 100))


def predict_with_confidence(
        model, study_hours: float, absences: int, focus: int
) -> tuple[float, float, list[float]]:
    """
    Return (score, confidence_pct, [lr_pred, rf_pred, gb_pred]).

    Confidence = how well the 3 base learners agree.
    High agreement → high confidence (up to 97%).
    Extreme / atypical inputs → slight penalty.
    """
    X        = build_row(study_hours, absences, focus)
    scaler   = model.named_steps["scaler"]
    stacking = model.named_steps["model"]
    Xs       = scaler.transform(X)

    indiv = [float(np.clip(est.predict(Xs)[0], 0, 100))
             for est in stacking.estimators_]

    score      = float(np.clip(model.predict(X)[0], 0, 100))
    std_dev    = np.std(indiv)
    # Base confidence from agreement
    conf = max(60.0, 95.0 - std_dev * 2.5)
    # Small penalty when inputs are far from typical training distribution
    dist = (abs(study_hours - 5.5) / 4.5 +
            abs(absences - 3) / 17 +
            abs(focus - 7) / 3) / 3
    conf = round(conf * (1 - dist * 0.08), 1)
    conf = min(conf, 97.0)
    return score, conf, indiv


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_level(score: float) -> tuple[str, str]:
    for lo, hi, name, col in LEVELS:
        if lo <= score < hi:
            return name, col
    return LEVELS[-1][2], LEVELS[-1][3]


# ── Weakness / bad-habit detection ───────────────────────────────────────────

def detect_weaknesses(study_hours: float, absences: int,
                      focus: int) -> list[dict]:
    """Return list of weakness cards with severity colour."""
    att = (40 - absences) / 40 * 100
    out = []

    if study_hours < 3:
        out.append({"sev": "🔴 Nghiêm trọng", "col": "card-red",
                    "title": "Thiếu thời gian tự học",
                    "body": f"Chỉ {study_hours}h/ngày — thấp hơn mức tối thiểu 4h.",
                    "fix" : "Tăng dần thêm 30 phút mỗi tuần cho đến khi đạt 5-6h."})
    elif study_hours < 5:
        out.append({"sev": "🟡 Trung bình", "col": "card-orange",
                    "title": "Thời gian học chưa đủ",
                    "body": f"Đang ở {study_hours}h/ngày, cần ít nhất 5h.",
                    "fix" : "Chặn 1 giờ cố định mỗi buổi sáng để học lý thuyết."})

    if absences > 10:
        out.append({"sev": "🔴 Nghiêm trọng", "col": "card-red",
                    "title": "Tỉ lệ vắng mặt quá cao",
                    "body": f"Nghỉ {absences} buổi ({100-att:.0f}% absent).",
                    "fix" : "Đặt lịch nhắc nhở buổi sáng; học với bạn để có động lực."})
    elif absences > 5:
        out.append({"sev": "🟡 Trung bình", "col": "card-orange",
                    "title": "Vắng mặt nhiều",
                    "body": f"Nghỉ {absences} buổi — bỏ lỡ nội dung quan trọng.",
                    "fix" : "Mục tiêu: không nghỉ buổi nào trong 3 tuần liền tới."})

    if focus < 4:
        out.append({"sev": "🔴 Nghiêm trọng", "col": "card-red",
                    "title": "Độ tập trung rất thấp",
                    "body": f"Chỉ {focus}/10 — nhiều giờ học bị lãng phí.",
                    "fix" : "Tắt mạng khi học, dùng app Forest/Cold Turkey chặn mạng."})
    elif focus < 6:
        out.append({"sev": "🟡 Trung bình", "col": "card-orange",
                    "title": "Tập trung chưa tốt",
                    "body": f"Mức tập trung {focus}/10 làm giảm hiệu quả.",
                    "fix" : "Thử Pomodoro 25 phút học / 5 phút nghỉ."})

    if not out:
        out.append({"sev": "🟢 Tốt", "col": "card-green",
                    "title": "Không phát hiện điểm yếu nghiêm trọng",
                    "body": "Thói quen học tập của bạn cân bằng tốt!",
                    "fix" : "Duy trì và hướng dẫn bạn bè cùng phương pháp."})
    return out


def detect_bad_habits(study_hours: float, absences: int,
                      focus: int) -> list[dict]:
    """Identify counter-productive behaviour patterns."""
    att = (40 - absences) / 40 * 100
    habits = []

    if study_hours > 8 and focus < 5:
        habits.append({
            "icon": "🔥", "name": "Kiệt sức (Burnout)",
            "desc": "Học quá nhiều giờ nhưng tập trung kém — não bộ đã bão hoà.",
            "advice": "Giảm xuống 6-7h/ngày và tăng chất lượng mỗi giờ."})

    if absences <= 2 and study_hours < 3:
        habits.append({
            "icon": "😴", "name": "Học thụ động",
            "desc": "Đi học đủ nhưng không tự học — kiến thức không đọng lại.",
            "advice": "Dành 1h ôn bài ngay sau mỗi buổi học trên lớp."})

    if focus >= 8 and att < 65:
        habits.append({
            "icon": "📵", "name": "Tự học cực đoan",
            "desc": "Tập trung cao nhưng bỏ lớp nhiều — mất định hướng từ giảng viên.",
            "advice": "Kết hợp tự học với tham dự lớp đầy đủ."})

    if study_hours >= 3 and absences == 0 and focus >= 7:
        habits.append({
            "icon": "⭐", "name": "Thói quen học tập lành mạnh",
            "desc": "Profile học tập tốt và cân bằng.",
            "advice": "Tiếp tục và nâng mức độ thử thách dần lên."})

    return habits


def sensitivity(model, study_hours: float,
                absences: int, focus: int) -> dict[str, float]:
    """Change in predicted score for ±1 unit of each input."""
    base = predict(model, study_hours, absences, focus)
    return {
        "📖 +1h học"        : predict(model, min(10, study_hours + 1), absences, focus) - base,
        "📖 −1h học"        : predict(model, max(1,  study_hours - 1), absences, focus) - base,
        "🏫 −2 buổi vắng"  : predict(model, study_hours, max(0,  absences - 2), focus) - base,
        "🏫 +2 buổi vắng"  : predict(model, study_hours, min(20, absences + 2), focus) - base,
        "🎯 +2 tập trung"  : predict(model, study_hours, absences, min(10, focus + 2)) - base,
        "🎯 −2 tập trung"  : predict(model, study_hours, absences, max(1,  focus - 2)) - base,
    }


def find_time_for_target(model, absences: int,
                          focus: int, target: float) -> int | None:
    """Return minimum study hours (1-10) to reach target score."""
    for h in range(1, 11):
        if predict(model, float(h), absences, focus) >= target:
            return h
    return None


def build_study_plan(score, study_hours, absences, focus) -> list[dict]:
    plan = []
    target = max(score + 10, 70)

    if study_hours < 4:
        plan.append({"area": "⏱️ Giờ học",
                     "action": f"Tăng từ {study_hours}h → 5h/ngày (thêm 30 phút/tuần)",
                     "p": "🔴"})
    elif study_hours < 6:
        plan.append({"area": "⏱️ Giờ học",
                     "action": "Tăng thêm 1h cuối tuần để bù khoảng trống kiến thức",
                     "p": "🟡"})
    else:
        plan.append({"area": "⏱️ Giờ học",
                     "action": "Xuất sắc — duy trì và đảm bảo chất lượng mỗi giờ",
                     "p": "🟢"})

    if absences > 6:
        plan.append({"area": "🏫 Chuyên cần",
                     "action": f"Giảm từ {absences} → tối đa 3 buổi vắng/kỳ",
                     "p": "🔴"})
    elif absences > 3:
        plan.append({"area": "🏫 Chuyên cần",
                     "action": "Không nghỉ buổi nào trong tháng tới",
                     "p": "🟡"})
    else:
        plan.append({"area": "🏫 Chuyên cần",
                     "action": "Tỉ lệ tham dự tốt — tiếp tục duy trì",
                     "p": "🟢"})

    if focus < 5:
        plan.append({"area": "🎯 Tập trung",
                     "action": "Pomodoro 25/5 + tắt điện thoại khi học",
                     "p": "🔴"})
    elif focus < 8:
        plan.append({"area": "🎯 Tập trung",
                     "action": "Tìm không gian yên tĩnh, nghe nhạc không lời",
                     "p": "🟡"})
    else:
        plan.append({"area": "🎯 Tập trung",
                     "action": "Tập trung tốt — chia sẻ phương pháp với bạn bè",
                     "p": "🟢"})

    if score < 55:
        plan.append({"area": "📚 Hỗ trợ",
                     "action": "Gặp giảng viên hỗ trợ + tham gia nhóm học 3x/tuần",
                     "p": "🔴"})
    elif score < 70:
        plan.append({"area": "📚 Luyện tập",
                     "action": "Làm thêm bài tập cũ, ôn chủ đề yếu mỗi cuối tuần",
                     "p": "🟡"})

    plan.append({"area": "😴 Sức khoẻ",
                 "action": "Ngủ 7–8h/đêm + tập 30 phút/ngày — não bộ hoạt động tốt hơn",
                 "p": "🟡"})
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ax_dark(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, alpha=0.12, color=GRID)
    return ax


def _fig(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(BG)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def tab_dashboard(model, study_hours, absences, focus,
                  score, conf, indiv, deriv):

    lv_name, lv_col = get_level(score)

    # ── Score hero ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="score-hero">
        <div class="score-num" style="color:{lv_col};">{score:.1f}</div>
        <div class="score-sub" style="color:{lv_col};">
            / 100 &nbsp;|&nbsp; {lv_name}
        </div>
    </div>""", unsafe_allow_html=True)

    # Progress + confidence side by side
    pc1, pc2 = st.columns([3, 1])
    pc1.progress(min(int(score), 100),
                 text=f"Điểm tổng kết dự đoán: {score:.1f} / 100")
    conf_col = "#2ecc71" if conf >= 85 else "#f1c40f" if conf >= 70 else "#e74c3c"
    pc2.markdown(f"""
    <div style="text-align:center;padding:8px;background:#1e2130;
         border-radius:10px;border:1px solid #30363d;">
        <span style="font-size:11px;opacity:.7">ĐỘ TIN CẬY</span><br>
        <span style="font-size:26px;font-weight:800;color:{conf_col};">{conf:.0f}%</span>
    </div>""", unsafe_allow_html=True)

    # ── 4 key metrics ─────────────────────────────────────────────────────────
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    diff = score - DATASET_MEAN
    m1.metric("🎯 Điểm dự đoán",  f"{score:.1f}")
    m2.metric("📊 vs Trung bình",
              f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}",
              delta_color="normal" if diff >= 0 else "inverse")
    m3.metric("🏫 Chuyên cần",    f"{deriv['attendance_rate']:.0f}%")
    m4.metric("⚡ Hiệu suất học", f"{deriv['study_efficiency']:.0f}%")

    st.divider()

    # ── Estimated performance features ───────────────────────────────────────
    st.markdown("#### 🔬 Đặc trưng được ước tính từ thói quen học")
    att_pct = deriv["attendance_rate"]
    est     = _estimate_hidden(float(study_hours), att_pct, focus)
    f1, f2, f3 = st.columns(3)
    f1.metric("📝 Điểm bài tập (ước tính)", f"{est['assignment_score']:.1f}")
    f2.metric("📋 Điểm giữa kỳ (ước tính)", f"{est['midterm_score']:.1f}")
    f3.metric("💪 Cường độ học",             f"{deriv['study_intensity']:.1f}")

    # ── Base learner agreement ────────────────────────────────────────────────
    with st.expander("🔍 Chi tiết dự đoán từng mô hình (minh bạch AI)"):
        names = ["Linear Regression", "Random Forest", "Gradient Boosting"]
        b1, b2, b3 = st.columns(3)
        for col, name, val in zip([b1, b2, b3], names, indiv):
            col.metric(name, f"{val:.1f}")
        st.caption(
            f"Stacking meta-learner (Ridge) kết hợp 3 dự đoán trên → {score:.1f}  |  "
            f"Độ lệch chuẩn: {np.std(indiv):.2f} điểm"
        )

    st.divider()

    # ── Level ladder ──────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Hệ thống cấp độ")
    lv_cols = st.columns(5)
    for i, (lo, hi, name, col) in enumerate(LEVELS):
        active = lo <= score < hi
        lv_cols[i].markdown(f"""
        <div style="background:{'%s22'%col if active else '#1a1a2e'};
             border:{'2px solid '+col if active else '1px solid '+GRID};
             border-radius:10px;padding:10px 4px;text-align:center;">
            <b style="color:{col};font-size:12px">{name}</b><br>
            <span style="opacity:.6;font-size:11px">{lo}–{hi} điểm</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Weakness cards ────────────────────────────────────────────────────────
    st.markdown("#### ⚠️ Phát hiện điểm yếu")
    for w in detect_weaknesses(study_hours, absences, focus):
        st.markdown(f"""
        <div class="card {w['col']}">
            <b>{w['sev']} — {w['title']}</b><br>
            {w['body']}<br>
            <span style="color:#3498db">💡 {w['fix']}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LEARNING ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

def tab_assistant(model, study_hours, absences, focus, score, target_score):

    # ── Goal tracking ─────────────────────────────────────────────────────────
    st.markdown("#### 🎯 Theo dõi mục tiêu")
    gap = target_score - score
    gap_col = "#2ecc71" if gap <= 0 else "#e67e22" if gap <= 10 else "#e74c3c"

    g1, g2, g3 = st.columns(3)
    g1.metric("Điểm hiện tại",  f"{score:.1f}")
    g2.metric("Mục tiêu",       f"{target_score}")
    g3.metric("Khoảng cách",
              f"+{gap:.1f}" if gap > 0 else f"{gap:.1f} (đã đạt!)",
              delta_color="inverse" if gap > 0 else "normal")

    # Progress towards goal
    progress_val = min(100, int(score / target_score * 100))
    st.progress(progress_val,
                text=f"Tiến độ đến mục tiêu {target_score}: {progress_val}%")

    if gap <= 0:
        st.success(f"🎉 Bạn đã đạt mục tiêu {target_score} điểm rồi! "
                   f"Hãy đặt mục tiêu cao hơn.")
    else:
        # Find what's needed to hit target
        h_needed = find_time_for_target(model, absences, focus, target_score)
        if h_needed and h_needed > study_hours:
            st.info(
                f"📌 Nếu tăng giờ học lên **{h_needed}h/ngày** "
                f"(giữ nguyên các yếu tố khác) → đạt **{target_score}** điểm.")
        elif h_needed is None:
            st.warning(
                "⚠️ Chỉ tăng giờ học chưa đủ. Cần cải thiện cả tập trung "
                "và chuyên cần để đạt mục tiêu này.")

    st.divider()

    # ── Time optimizer ────────────────────────────────────────────────────────
    st.markdown("#### ⏱️ Tối ưu thời gian học")
    hours_range = np.arange(1, 10.5, 0.5)
    scores_h    = [predict(model, h, absences, focus) for h in hours_range]
    opt_idx     = int(np.argmax(scores_h))
    opt_h       = hours_range[opt_idx]

    fig, ax = _fig(figsize=(10, 3.8))
    ax = fig.axes[0]; _ax_dark(ax)

    ax.plot(hours_range, scores_h, color="#2ecc71", lw=2.5, zorder=3)
    ax.fill_between(hours_range, scores_h, alpha=0.13, color="#2ecc71")
    ax.axvline(study_hours, color="#e74c3c", ls="--", lw=1.8,
               label=f"Hiện tại: {study_hours}h → {score:.1f} đ")
    ax.axvline(opt_h, color="#9b59b6", ls="--", lw=1.4,
               label=f"Tối ưu: {opt_h}h → {max(scores_h):.1f} đ")
    ax.axhline(target_score, color="#f39c12", ls=":", lw=1.3,
               label=f"Mục tiêu: {target_score} đ")
    ax.axhline(DATASET_MEAN, color="grey", ls="--", lw=1, alpha=0.6,
               label=f"TB dataset: {DATASET_MEAN:.0f} đ")
    ax.set(xlabel="Giờ học / ngày", ylabel="Điểm dự đoán",
           title="Điểm dự đoán theo số giờ học (với absences & focus hiện tại)",
           ylim=(30, 100))
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    if opt_h > study_hours:
        gain = max(scores_h) - score
        st.info(
            f"💡 Tăng từ **{study_hours}h → {opt_h}h/ngày** có thể cải thiện điểm "
            f"thêm **+{gain:.1f} điểm** (giữ nguyên focus={focus} & absences={absences}).")

    st.divider()

    # ── Personalised study plan ───────────────────────────────────────────────
    st.markdown("#### 📋 Kế hoạch cải thiện cá nhân hoá")
    plan = build_study_plan(score, study_hours, absences, focus)
    for item in plan:
        p_col = {"🔴": "#e74c3c", "🟡": "#f1c40f", "🟢": "#2ecc71"}.get(item["p"], "#3498db")
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {p_col};">
            <b>{item['area']}</b>&emsp;
            <span style="font-size:12px;opacity:.75">{item['p']} Ưu tiên</span><br>
            {item['action']}
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Weekly schedule ───────────────────────────────────────────────────────
    st.markdown("#### 📆 Lịch học mẫu hàng tuần")
    target_h = max(4, int(study_hours) + (1 if score < target_score else 0))
    sched = pd.DataFrame({
        "Ngày"     : ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"],
        "Giờ học"  : [f"{target_h}h", f"{target_h}h", f"{target_h}h",
                      f"{target_h}h", f"{target_h}h", f"{target_h+1}h", "Tuỳ chọn"],
        "Nội dung" : ["Lý thuyết mới", "Bài tập + ôn", "Lý thuyết mới",
                      "Bài tập + ôn", "Tổng hợp tuần", "Ôn sâu chủ đề yếu",
                      "Nghỉ / nhóm học"],
        "Mục tiêu" : [f"Hiểu {target_h*10}% chương mới", "Làm đúng 80% bài",
                      f"Hiểu {target_h*10}% chương mới", "Làm đúng 80% bài",
                      "Không có lỗ hổng tuần này", "Tăng điểm yếu ≥5 điểm",
                      "Nạp lại năng lượng"],
    })
    st.dataframe(sched, use_container_width=True, hide_index=True)

    # ── CSV Export ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📥 Tải xuống báo cáo")
    lv_name, _ = get_level(score)
    att_pct    = max(40.0, (40 - absences) / 40 * 100)
    est        = _estimate_hidden(float(study_hours), att_pct, focus)
    export = pd.DataFrame([{
        "Giờ học/ngày"         : study_hours,
        "Số buổi vắng"         : absences,
        "Độ tập trung (1-10)"  : focus,
        "Tỉ lệ tham dự (%)"    : att_pct,
        "Điểm BT ước tính"     : round(est["assignment_score"], 1),
        "Điểm GK ước tính"     : round(est["midterm_score"], 1),
        "Điểm dự đoán"         : round(score, 2),
        "Xếp loại"             : lv_name,
        "Mục tiêu"             : target_score,
        "Khoảng cách mục tiêu" : round(target_score - score, 2),
        **{f"KH - {p['area']}": p["action"] for p in plan},
    }])
    buf = io.StringIO()
    export.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Tải CSV báo cáo", buf.getvalue().encode("utf-8-sig"),
        "learning_report.csv", "text/csv",
        type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AI COACH
# ─────────────────────────────────────────────────────────────────────────────

def tab_coach(model, study_hours, absences, focus, score):

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("#### 🕸️ Biểu đồ radar — Hồ sơ học tập")

    att_pct = max(40.0, (40 - absences) / 40 * 100)
    est     = _estimate_hidden(float(study_hours), att_pct, focus)

    # 5 dimensions, each normalised 0-1
    dims = ["Giờ học", "Chuyên cần", "Tập trung", "Bài tập", "Giữa kỳ"]
    user_vals = [
        (study_hours - 1) / 9,
        (att_pct - 40) / 60,
        (focus - 1) / 9,
        (est["assignment_score"] - 30) / 70,
        (est["midterm_score"] - 25) / 75,
    ]
    ideal_vals = [1.0, 1.0, 1.0, 1.0, 1.0]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    user_vals += user_vals[:1]
    ideal_vals += ideal_vals[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.plot(angles, ideal_vals, "o--", color="#30363d",  lw=1.5, alpha=0.5)
    ax.fill(angles, ideal_vals, color="#30363d", alpha=0.10)
    ax.plot(angles, user_vals,  "o-",  color="#3498db",  lw=2.5)
    ax.fill(angles, user_vals,  color="#3498db", alpha=0.22)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=WHITE, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                       color="grey", fontsize=8)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, alpha=0.4)
    ax.set_title("Hồ sơ học tập (xanh=bạn / xám=lý tưởng)",
                 color=WHITE, fontsize=11, pad=18)

    patch_user  = mpatches.Patch(color="#3498db", label="Hồ sơ của bạn")
    patch_ideal = mpatches.Patch(color="#30363d", label="Lý tưởng")
    ax.legend(handles=[patch_user, patch_ideal],
              facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE,
              loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    col_r, col_s = st.columns([1, 1])
    col_r.pyplot(fig); plt.close(fig)

    # Radar scores summary
    score_map = {
        "Giờ học"  : user_vals[0],
        "Chuyên cần": user_vals[1],
        "Tập trung": user_vals[2],
        "Bài tập"  : user_vals[3],
        "Giữa kỳ" : user_vals[4],
    }
    weakest = min(score_map, key=score_map.get)
    with col_s:
        st.markdown(f"""
        <div class="card card-blue" style="margin-top:30px;">
            <b>📊 Điểm số từng chiều (% so với lý tưởng)</b><br><br>
            {'<br>'.join(
                f'{"✅" if v >= 0.7 else "⚠️" if v >= 0.4 else "❌"} '
                f'<b>{k}</b>: {v*100:.0f}%'
                for k, v in score_map.items()
            )}<br><br>
            🎯 <b>Điểm yếu nhất: {weakest}</b> — ưu tiên cải thiện trước.
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Bad-habit detection ───────────────────────────────────────────────────
    st.markdown("#### 🚨 Phát hiện thói quen xấu")
    for h in detect_bad_habits(study_hours, absences, focus):
        bd_col = "card-green" if h["icon"] == "⭐" else "card-orange"
        st.markdown(f"""
        <div class="card {bd_col}">
            <b>{h['icon']} {h['name']}</b><br>
            {h['desc']}<br>
            <span style="color:#3498db">✅ {h['advice']}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── 7-day coaching plan ───────────────────────────────────────────────────
    st.markdown("#### 📅 Kế hoạch huấn luyện 7 ngày")
    tips_by_score = {
        "low"  : [
            ("Ngày 1-2", "🎯", "Đánh giá lỗ hổng kiến thức", "Làm 1 đề cũ và ghi lại bài nào sai nhiều nhất."),
            ("Ngày 3-4", "📖", "Tập trung vào chủ đề yếu",   "Học lại từ đầu chủ đề sai nhiều; không học cái mới."),
            ("Ngày 5",   "👥", "Học nhóm",                   "Giải thích lại cho bạn bè — cách tốt nhất để nhớ."),
            ("Ngày 6",   "📝", "Làm bài kiểm tra thử",       "Thi thử 45 phút, không xem tài liệu."),
            ("Ngày 7",   "😴", "Nghỉ ngơi hoàn toàn",        "Não cần thời gian củng cố — đừng học thêm."),
        ],
        "mid"  : [
            ("Ngày 1-2", "📖", "Đọc trước chủ đề tuần tới", "Xem qua bài trước khi lên lớp → hiểu nhanh hơn 2x."),
            ("Ngày 3-4", "🧩", "Làm bài tập nâng cao",      "Chọn bài khó hơn mức bình thường một bậc."),
            ("Ngày 5",   "🎯", "Ôn lại điểm yếu",           "30 phút/ngày ôn chủ đề đang dưới 60%."),
            ("Ngày 6",   "📊", "Tự đánh giá",               "Viết 5 điều đã học tuần này mà không nhìn sách."),
            ("Ngày 7",   "🏃", "Hoạt động ngoài trời",      "Tập thể dục giúp não consolidate kiến thức khi ngủ."),
        ],
        "high" : [
            ("Ngày 1-2", "🚀", "Thách thức bản thân",        "Đọc thêm tài liệu nâng cao, báo cáo khoa học."),
            ("Ngày 3-4", "🤝", "Mentor bạn bè",              "Dạy bạn bè — deepens your own understanding."),
            ("Ngày 5",   "🏆", "Tham gia cuộc thi / project", "Áp dụng kiến thức vào dự án thực tế."),
            ("Ngày 6",   "📝", "Viết tóm tắt kiến thức",     "Tạo mind-map hoặc note Notion tóm tắt tuần."),
            ("Ngày 7",   "😴", "Nghỉ ngơi chất lượng",       "Đọc sách không liên quan học tập — reset não."),
        ],
    }
    tier_key = "low" if score < 55 else "high" if score >= 75 else "mid"
    coach_df = pd.DataFrame(tips_by_score[tier_key],
                             columns=["Thời điểm", "Icon", "Hoạt động", "Chi tiết"])
    for _, row in coach_df.iterrows():
        st.markdown(f"""
        <div class="coach-tip">
            <b>{row['Thời điểm']} &nbsp; {row['Icon']} {row['Hoạt động']}</b><br>
            <span style="opacity:.85">{row['Chi tiết']}</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def tab_analysis(model, study_hours, absences, focus, score):

    # ── Sensitivity bar chart ─────────────────────────────────────────────────
    st.markdown("#### 📐 Phân tích độ nhạy — Mỗi thay đổi ảnh hưởng bao nhiêu?")
    delta  = sensitivity(model, study_hours, absences, focus)
    labels = list(delta.keys())
    values = list(delta.values())
    cols   = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = _fig(figsize=(9, 4)); ax = fig.axes[0]; _ax_dark(ax)
    bars = ax.barh(labels, values, color=cols, edgecolor=BG)
    ax.axvline(0, color=WHITE, lw=1)
    ax.set(xlabel="Thay đổi điểm số",
           title="Tác động của từng điều chỉnh thói quen (+/−)")
    for b, v in zip(bars, values):
        s = "+" if v >= 0 else ""
        ax.text(v + (0.05 if v >= 0 else -0.05), b.get_y() + b.get_height() / 2,
                f"{s}{v:.2f}", va="center",
                ha="left" if v >= 0 else "right",
                color=WHITE, fontsize=10, fontweight="bold")
    plt.setp(ax.get_yticklabels(), color=WHITE)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()

    # ── Score vs hours + focus bar ────────────────────────────────────────────
    st.markdown("#### 📈 Đường cong điểm số")
    h_arr = np.arange(1, 10.5, 0.5)
    f_arr = list(range(1, 11))
    s_h   = [predict(model, h, absences, focus) for h in h_arr]
    s_f   = [predict(model, study_hours, absences, f) for f in f_arr]

    fig, (ax1, ax2) = _fig(1, 2, figsize=(13, 4.5))
    _ax_dark(ax1); _ax_dark(ax2)

    ax1.plot(h_arr, s_h, color="#2ecc71", lw=2.5, zorder=3)
    ax1.fill_between(h_arr, s_h, alpha=0.12, color="#2ecc71")
    ax1.axvline(study_hours, color="#e74c3c", ls="--", lw=1.8,
                label=f"Bạn: {study_hours}h → {score:.1f}")
    ax1.axhline(DATASET_MEAN, color="grey", ls="--", lw=1, alpha=0.7,
                label=f"TB: {DATASET_MEAN:.0f}")
    ax1.set(xlabel="Giờ học / ngày", ylabel="Điểm",
            title="Điểm vs Giờ học", ylim=(30, 100))
    ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    bcols = ["#e74c3c" if f < 5 else "#f39c12" if f < 8 else "#2ecc71" for f in f_arr]
    ax2.bar(f_arr, s_f, color=bcols, edgecolor=BG, lw=0.8)
    ax2.axvline(focus, color=WHITE, ls="--", lw=1.5, alpha=0.7,
                label=f"Bạn: {focus}/10")
    ax2.axhline(DATASET_MEAN, color="grey", ls="--", lw=1, alpha=0.7,
                label=f"TB: {DATASET_MEAN:.0f}")
    ax2.set(xlabel="Độ tập trung (1-10)", ylabel="Điểm",
            title="Điểm vs Tập trung", ylim=(30, 100))
    ax2.set_xticks(f_arr)
    ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, fontsize=9)

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Bản đồ nhiệt — Giờ học × Độ tập trung")
    h_vals = list(range(1, 11)); f_vals = list(range(1, 11))
    grid   = np.array([
        [predict(model, float(h), absences, f) for h in h_vals]
        for f in reversed(f_vals)
    ])
    fig2, ax2h = _fig(figsize=(9, 5)); ax2h = fig2.axes[0]; _ax_dark(ax2h)
    im = ax2h.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=40, vmax=90)
    ax2h.set_xticks(range(10)); ax2h.set_xticklabels(h_vals, color=WHITE)
    ax2h.set_yticks(range(10)); ax2h.set_yticklabels(list(reversed(f_vals)), color=WHITE)
    ax2h.set(xlabel="Giờ học / ngày", ylabel="Độ tập trung",
             title="Điểm dự đoán (★ = vị trí của bạn)")
    try:
        ux = h_vals.index(int(study_hours))
        uy = list(reversed(f_vals)).index(focus)
        ax2h.plot(ux, uy, "w*", ms=18)
    except ValueError:
        pass
    cbar = fig2.colorbar(im, ax=ax2h)
    plt.setp(cbar.ax.get_yticklabels(), color=WHITE)
    cbar.ax.yaxis.set_tick_params(color=WHITE)
    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.divider()

    # ── Interactive What-If ───────────────────────────────────────────────────
    st.markdown("#### 🔮 Thử nghiệm kịch bản What-If")
    wc1, wc2, wc3 = st.columns(3)
    wh = wc1.slider("⏱️ Giờ học mới",   1, 10, int(study_hours), key="wh")
    wa = wc2.slider("🏫 Nghỉ học mới",  0, 20, absences,         key="wa")
    wf = wc3.slider("🎯 Tập trung mới", 1, 10, focus,            key="wf")

    ws   = predict(model, float(wh), wa, wf)
    diff = ws - score
    d_col = "#2ecc71" if diff >= 0 else "#e74c3c"
    st.markdown(f"""
    <div class="card" style="text-align:center;padding:22px;">
        <span style="font-size:15px;opacity:.7">Điểm hiện tại</span><br>
        <span style="font-size:40px;font-weight:900;">{score:.1f}</span>
        &nbsp;
        <span style="font-size:22px;color:{d_col}">
            {'▲' if diff >= 0 else '▼'}
        </span>
        &nbsp;
        <span style="font-size:40px;font-weight:900;color:{d_col};">{ws:.1f}</span>
        <br>
        <span style="color:{d_col};font-size:16px;font-weight:700;">
            {'+' if diff >= 0 else ''}{diff:.1f} điểm so với hiện tại
        </span>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Comparison chart ──────────────────────────────────────────────────────
    st.markdown("#### 🏆 So sánh với các nhóm sinh viên")
    fig3, ax3 = _fig(figsize=(7, 3.8)); ax3 = fig3.axes[0]; _ax_dark(ax3)
    cats  = ["Bạn", "Trung bình", "Top 25%"]
    cvals = [score, DATASET_MEAN, DATASET_TOP25]
    bcs   = ["#3498db", "#9b59b6", "#2ecc71"]
    bars3 = ax3.bar(cats, cvals, color=bcs, edgecolor=BG, width=0.45)
    for b, v in zip(bars3, cvals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 f"{v:.1f}", ha="center", va="bottom",
                 color=WHITE, fontweight="bold", fontsize=13)
    ax3.set(ylabel="Điểm", title="Bạn vs Trung bình vs Top 25%", ylim=(0, 105))
    plt.setp(ax3.get_xticklabels(), color=WHITE)
    plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AI Learning Habit Analyzer",
        page_icon="🎓", layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 AI Learning Coach")
        st.markdown("Hệ thống phân tích thói quen học tập & dự đoán kết quả.")
        st.divider()

        st.markdown("### 📋 Thói quen học tập")
        study_hours  = st.slider("⏱️ Giờ học mỗi ngày",       1, 10, 5,
                                 help="Số giờ tự học trung bình mỗi ngày")
        absences     = st.slider("🏫 Số buổi vắng mặt",       0, 20,  3,
                                 help="Tổng số buổi nghỉ học trong kỳ (40 buổi/kỳ)")
        focus_score  = st.slider("🎯 Độ tập trung (1–10)",     1, 10,  7,
                                 help="1=rất phân tâm, 10=cực kỳ tập trung")
        st.divider()

        st.markdown("### 🎯 Đặt mục tiêu")
        target_score = st.slider("Điểm mục tiêu", 50, 100, 80,
                                 help="Điểm bạn muốn đạt cuối kỳ")
        st.divider()

        run_btn = st.button("🚀 Phân tích ngay",
                            type="primary", use_container_width=True)
        st.divider()
        st.markdown("""
        <div style="font-size:11px;opacity:.5;text-align:center;line-height:1.8">
            Stacking: LR + RF + Gradient Boosting<br>
            Dataset: 10,000 sinh viên<br>
            MAE ≈ 6.2 điểm &nbsp;|&nbsp; R² ≈ 0.50
        </div>""", unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎓 AI-based Learning Habit Analysis System")
    st.markdown(
        "Nhập thông tin học tập ở sidebar → nhấn **🚀 Phân tích ngay** "
        "để xem dự đoán, phân tích và kế hoạch cải thiện cá nhân hoá."
    )

    # ── Landing page ──────────────────────────────────────────────────────────
    if not run_btn:
        st.info("👈 Kéo thanh trượt bên trái và nhấn **Phân tích ngay** để bắt đầu.")
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, title, desc in [
            (c1, "🤖", "3 Mô hình ML",         "LR + RF + Gradient Boosting → Stacking"),
            (c2, "📚", "Learning Assistant",    "Goal tracking + time optimizer + plan"),
            (c3, "🧠", "AI Coach",              "Radar chart + bad-habit + coaching tips"),
            (c4, "📊", "Phân tích chuyên sâu", "What-if + heatmap + comparison"),
        ]:
            col.markdown(f"""
            <div class="feat-box">
                <div style="font-size:36px;margin-bottom:8px">{icon}</div>
                <b>{title}</b><br>
                <span style="opacity:.6;font-size:13px">{desc}</span>
            </div>""", unsafe_allow_html=True)
        return

    # ── Compute predictions ───────────────────────────────────────────────────
    with st.spinner("Đang phân tích…"):
        model            = load_model()
        score, conf, ind = predict_with_confidence(
            model, float(study_hours), absences, focus_score)
        deriv            = derived_metrics(float(study_hours), absences, focus_score)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🎯 Dashboard",
        "📚 Learning Assistant",
        "🤖 AI Coach",
        "📊 Phân tích",
    ])

    with tabs[0]:
        tab_dashboard(model, float(study_hours), absences, focus_score,
                      score, conf, ind, deriv)
    with tabs[1]:
        tab_assistant(model, float(study_hours), absences, focus_score,
                      score, target_score)
    with tabs[2]:
        tab_coach(model, float(study_hours), absences, focus_score, score)
    with tabs[3]:
        tab_analysis(model, float(study_hours), absences, focus_score, score)


if __name__ == "__main__":
    main()
