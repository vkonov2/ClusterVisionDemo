// frontend/src/components/SiteFooter.tsx
import React from "react";
import { Link } from "react-router-dom";
import logo from "../assets/logo.png";

export default function SiteFooter() {
  return (
    <footer className="mx-auto max-w-7xl px-5 py-12 text-gray-300 text-sm">
      <div className="grid md:grid-cols-3 gap-8 border-t border-white/10 pt-8">
        <div>
          <Link to="/" className="inline-flex items-center gap-2 mb-3">
            {/* Увеличил ~в 1.5 раза */}
            <img src={logo} alt="SynapSight" className="h-16 w-auto md:h-20" />
          </Link>
          <p className="text-gray-400 text-sm mb-3 max-w-sm">
            Платформа претестирования видеорекламы с AI-анализом креатива и прогнозом нейроэффективности.
          </p>
          <p className="text-xs opacity-80">© {new Date().getFullYear()} SynapSight. Все права защищены.</p>
        </div>
        <div>
          <h5 className="font-semibold mb-2">Меню</h5>
          <ul className="space-y-1">
            <li><a href="#upload" className="hover:text-white">Продукт</a></li>
            <li><a href="#features" className="hover:text-white">Как это работает</a></li>
            <li><a href="#metrics" className="hover:text-white">Метрики</a></li>
            <li><a href="#pricing" className="hover:text-white">Тарифы</a></li>
            <li><a href="#faq" className="hover:text-white">FAQ</a></li>
          </ul>
        </div>
        <div>
          <h5 className="font-semibold mb-2">Контакты</h5>
          <p className="text-xs mb-1">Email: <a href="mailto:hello@synapsight.io" className="underline">hello@synapsight.io</a></p>
          <p className="text-xs mb-1">Telegram: @synapsight</p>
          <p className="text-xs opacity-70">ООО «СинапСайт» · ОГРН ХХХХХХХХХХХХ · ИНН ХХХХХХХХХ</p>
        </div>
      </div>
    </footer>
  );
}
