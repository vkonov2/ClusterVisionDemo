// frontend/src/components/SiteHeader.tsx
import React from "react";
import { Link } from "react-router-dom";
// импортируем файл из src/assets
import logo from "../assets/logo.png";

type NavItem = { label: string; href: string };
export default function SiteHeader({
  nav = [],
  rightSlot,
}: { nav?: NavItem[]; rightSlot?: React.ReactNode }) {
  return (
    <header className="sticky top-0 z-30 backdrop-blur bg-[rgba(21,19,29,0.8)] h-[100px] flex items-center">
      <div className="mx-auto max-w-7xl px-6 flex items-center justify-between w-full h-full">
        <div className="flex items-center gap-3 whitespace-nowrap">
          <Link to="/" className="inline-flex items-center gap-2" aria-label="Главная">
            {/* Увеличил ~в 1.5 раза */}
            <img src={logo} alt="SynapSight" className="h-16 w-auto md:h-20" />
          </Link>
        </div>
        <nav className="hidden lg:flex items-center gap-8 text-sm text-gray-200/85">
          {nav.map((n) => (
            <a key={n.href} href={n.href} className="hover:text-white transition-colors">
              {n.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-3">{rightSlot}</div>
      </div>
    </header>
  );
}
